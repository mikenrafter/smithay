// The caching logic is used to process surface synchronization. It creates
// an effective decoupling between the moment the client sends wl_surface.commit
// and the moment where the state that was committed is actually applied by the
// compositor.
//
// The way this is modelled in Smithay is through the `Cache` type, which is a container
// representing a cached state for a particular type. The full cached state of a surface
// is thus composed of a a set of `Cache<T>` for all relevant `T`, as modelled by the
// `MultiCache`.
//
// The logic of the `Cache` is as follows:
//
// - The protocol handlers mutably access the `pending` state to modify it accord to
//   the client requests
// - On commit, a snapshot of this pending state is created by invoking `Cacheable::commit`
//   and stored in the cache alongside an externally provided id
// - When the compositor decides that a given state (represented by its commit id) should
//   become active, `Cache::apply_state` is invoked with that commit id. The associated state
//   is then applied to the `current` state, that the compositor can then use as a reference
//   for the current window state. Note that, to preserve the commit ordering, all states
//   with a commit id older than the one requested are applied as well, in order.
//
// The logic for generating these commit ids and deciding when to apply them is implemented
// and described in `transaction.rs`.

use std::{
    collections::VecDeque,
    sync::{Mutex, MutexGuard},
};

use downcast_rs::{Downcast, impl_downcast};
use wayland_server::DisplayHandle;

use crate::utils::Serial;

/// Trait representing a value that can be used in double-buffered storage
///
/// The type needs to implement the [`Default`] trait, which will be used
/// to initialize. You further need to provide two methods:
/// [`Cacheable::commit`] and [`Cacheable::merge_into`].
///
/// Double-buffered state works by having a "pending" instance of your type,
/// into which new values provided by the client are inserted. When the client
/// sends `wl_surface.commit`, the [`Cacheable::commit`] method will be
/// invoked on your value. This method is expected to produce a new instance of
/// your type, that will be stored in the cache, and eventually merged into the
/// current state.
///
/// In most cases, this method will simply produce a copy of the pending state,
/// but you might need additional logic in some cases, such as for handling
/// non-cloneable resources (which thus need to be moved into the produce value).
///
/// Then at some point the [`Cacheable::merge_into`] method of your type will be
/// invoked. In this method, `self` acts as the update that should be merged into
/// the current state provided as argument. In simple cases, the action would just
/// be to copy `self` into the current state, but more complex cases require
/// additional logic.
pub trait Cacheable: Default {
    /// Ordering priority when applying cached state; higher priorities apply first.
    const APPLY_PRIORITY: i32 = 0;
    /// Ordering priority when discarding cached state; higher priorities discard first.
    const DISCARD_PRIORITY: i32 = 0;

    /// Produce a new state to be cached from the pending state
    fn commit(&mut self, dh: &DisplayHandle) -> Self;
    /// Merge a state update into the current state
    fn merge_into(self, into: &mut Self, dh: &DisplayHandle);
    /// Discard a cached state update that will never become current.
    fn discard(self, _current: &mut Self) {}
    /// Discard a cached state update with access to retained queued states.
    ///
    /// This lets state that owns protocol resources avoid releasing a resource
    /// while an older or newer queued state still references it.
    fn discard_with_retained(self, current: &mut Self, _retained: &[&Self]) {
        self.discard(current);
    }
}

/// Double buffered cached state of type `T`
#[derive(Debug)]
pub struct CachedState<T> {
    pending: T,
    cache: VecDeque<(Serial, T)>,
    current: T,
}

impl<T: Default> Default for CachedState<T> {
    fn default() -> Self {
        CachedState {
            pending: T::default(),
            cache: VecDeque::new(),
            current: T::default(),
        }
    }
}

impl<T> CachedState<T> {
    /// Access the current state for `T`
    pub fn current(&mut self) -> &mut T {
        &mut self.current
    }

    /// Access the current state for `T` immutably.
    #[allow(dead_code)]
    pub(crate) fn current_ref(&self) -> &T {
        &self.current
    }

    /// Access the pending state for `T`
    pub fn pending(&mut self) -> &mut T {
        &mut self.pending
    }

    /// Iterate over queued cached states for `T`.
    #[allow(dead_code)]
    pub(crate) fn cached(&self) -> impl Iterator<Item = &T> {
        self.cache.iter().map(|(_, state)| state)
    }
}

trait Cache: Downcast {
    fn commit(&self, commit_id: Option<Serial>, dh: &DisplayHandle);
    fn apply_state(&self, commit_id: Serial, dh: &DisplayHandle);
    fn discard_state_range(&self, start_id: Serial, end_id: Serial);
    fn discard_all_states(&self);
    fn apply_priority(&self) -> i32;
    fn discard_priority(&self) -> i32;
}

impl_downcast!(Cache);

impl<T: Cacheable + 'static> Cache for Mutex<CachedState<T>> {
    fn commit(&self, commit_id: Option<Serial>, dh: &DisplayHandle) {
        let mut guard = self.lock().unwrap();
        let me = &mut *guard;
        let new_state = me.pending.commit(dh);
        if let Some(id) = commit_id {
            match me.cache.back_mut() {
                Some(&mut (cid, ref mut state)) if cid == id => new_state.merge_into(state, dh),
                _ => me.cache.push_back((id, new_state)),
            }
        } else {
            for (_, state) in me.cache.drain(..) {
                state.merge_into(&mut me.current, dh);
            }
            new_state.merge_into(&mut me.current, dh);
        }
    }

    fn apply_state(&self, commit_id: Serial, dh: &DisplayHandle) {
        let mut me = self.lock().unwrap();
        loop {
            if me.cache.front().map(|&(s, _)| s > commit_id).unwrap_or(true) {
                // if the cache is empty or the next state has a commit_id greater than the requested one
                break;
            }
            me.cache.pop_front().unwrap().1.merge_into(&mut me.current, dh);
        }
    }

    fn discard_state_range(&self, start_id: Serial, end_id: Serial) {
        let mut me = self.lock().unwrap();
        let mut index = 0;
        while index < me.cache.len() {
            let id = me.cache[index].0;
            if id < start_id {
                index += 1;
            } else if id <= end_id {
                let state = me.cache.remove(index).unwrap().1;
                let CachedState { current, cache, .. } = &mut *me;
                let retained = cache.iter().map(|(_, state)| state).collect::<Vec<_>>();
                state.discard_with_retained(current, &retained);
            } else {
                break;
            }
        }
    }

    fn discard_all_states(&self) {
        let mut me = self.lock().unwrap();
        let mut states = me.cache.drain(..).map(|(_, state)| state).collect::<Vec<_>>();
        while !states.is_empty() {
            let state = states.remove(0);
            let retained = states.iter().collect::<Vec<_>>();
            state.discard_with_retained(&mut me.current, &retained);
        }
    }

    fn apply_priority(&self) -> i32 {
        T::APPLY_PRIORITY
    }

    fn discard_priority(&self) -> i32 {
        T::DISCARD_PRIORITY
    }
}

/// A typemap-like container for double-buffered values
///
/// All values inserted into this container must implement the [`Cacheable`] trait,
/// which defines their buffering semantics. They furthermore must be [`Send`] as the surface state
/// can be accessed from multiple threads (but [`Sync`] is not required, the surface internally synchronizes
/// access to its state).
///
/// [`MultiCache::get`] provides access to the [`CachedState`] associated with a particular type.
///
/// Consumers of surface state (like compositor applications using Smithay) will mostly be concerned
/// with the [`CachedState::current`] method, which gives access to the current state of the surface for
/// a particular type.
///
/// Writers of protocol extensions logic will mostly be concerned with the [`CachedState::pending`] method,
/// which provides access to the pending state of the surface, in which new state from clients will be
/// stored.
///
/// This contained has [`Mutex`]-like semantics: values of multiple stored types can be accessed at the
/// same time, but accessing the same value multiple times will cause a deadlock.
/// The stored values are initialized lazily the first time [`get`][Self::get] is invoked with this type as argument.
pub struct MultiCache {
    caches: appendlist::AppendList<Box<dyn Cache + Send>>,
}

impl std::fmt::Debug for MultiCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiCache").finish_non_exhaustive()
    }
}

impl MultiCache {
    pub(crate) fn new() -> Self {
        Self {
            caches: appendlist::AppendList::new(),
        }
    }

    fn find_or_insert<T: Cacheable + Send + 'static>(&self) -> &Mutex<CachedState<T>> {
        for cache in &self.caches {
            if let Some(v) = (**cache).as_any().downcast_ref() {
                return v;
            }
        }
        // if we reach here, then the value is not yet in the list, insert it
        self.caches
            .push(Box::new(Mutex::new(CachedState::<T>::default())) as Box<_>);
        (*self.caches[self.caches.len() - 1])
            .as_any()
            .downcast_ref()
            .unwrap()
    }

    /// Access the [`CachedState`] associated with type `T`
    pub fn get<T: Cacheable + Send + 'static>(&self) -> MutexGuard<'_, CachedState<T>> {
        self.find_or_insert::<T>().lock().unwrap()
    }

    /// Check if the container currently contains values for type `T`
    pub fn has<T: Cacheable + Send + 'static>(&self) -> bool {
        self.caches
            .iter()
            .any(|c| (**c).as_any().is::<Mutex<CachedState<T>>>())
    }

    /// Commits the pending state, invoking Cacheable::commit()
    ///
    /// If commit_id is None, then the pending state is directly merged
    /// into the current state. Otherwise, this id is used to store the
    /// cached state. An id ca no longer be re-used as soon as not new id
    /// has been used in between. Provided IDs are expected to be provided
    /// in increasing order according to `Serial` semantics.
    ///
    /// If a None commit is given but there are some cached states, they'll
    /// all be merged into the current state before merging the pending one.
    pub(crate) fn commit(&mut self, commit_id: Option<Serial>, dh: &DisplayHandle) {
        // none of the underlying borrow_mut() can panic, as we hold
        // a &mut reference to the container, non are borrowed.
        let mut caches: Vec<_> = self.caches.iter().collect();
        caches.sort_by_key(|cache| std::cmp::Reverse((**cache).apply_priority()));
        for cache in caches {
            cache.commit(commit_id, dh);
        }
    }

    /// Apply given identified cached state to the current one
    ///
    /// All other preceding states are applied as well, to preserve commit ordering
    pub(crate) fn apply_state(&self, commit_id: Serial, dh: &DisplayHandle) {
        // none of the underlying borrow_mut() can panic, as we hold
        // a &mut reference to the container, non are borrowed.
        let mut caches: Vec<_> = self.caches.iter().collect();
        caches.sort_by_key(|cache| std::cmp::Reverse((**cache).apply_priority()));
        for cache in caches {
            cache.apply_state(commit_id, dh);
        }
    }

    /// Discard queued cached states with ids in the inclusive range.
    pub(crate) fn discard_state_range(&self, start_id: Serial, end_id: Serial) {
        let mut caches: Vec<_> = self.caches.iter().collect();
        caches.sort_by_key(|cache| std::cmp::Reverse((**cache).discard_priority()));
        for cache in caches {
            cache.discard_state_range(start_id, end_id);
        }
    }

    /// Discard all queued cached states without changing current state.
    pub(crate) fn discard_all_states(&self) {
        let mut caches: Vec<_> = self.caches.iter().collect();
        caches.sort_by_key(|cache| std::cmp::Reverse((**cache).discard_priority()));
        for cache in caches {
            cache.discard_all_states();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    static DISCARD_TEST_LOCK: Mutex<()> = Mutex::new(());
    static DISCARD_OBSERVATIONS: Mutex<Vec<(u32, Vec<u32>)>> = Mutex::new(Vec::new());

    #[derive(Default)]
    struct DiscardProbe {
        id: u32,
    }

    #[derive(Default)]
    struct ReleaseProbe {
        buffer_id: Option<u32>,
        releases: Option<Arc<AtomicUsize>>,
    }

    impl Cacheable for DiscardProbe {
        fn commit(&mut self, _dh: &DisplayHandle) -> Self {
            Self { id: self.id }
        }

        fn merge_into(self, into: &mut Self, _dh: &DisplayHandle) {
            into.id = self.id;
        }

        fn discard_with_retained(self, _current: &mut Self, retained: &[&Self]) {
            DISCARD_OBSERVATIONS
                .lock()
                .unwrap()
                .push((self.id, retained.iter().map(|state| state.id).collect()));
        }
    }

    impl Cacheable for ReleaseProbe {
        fn commit(&mut self, _dh: &DisplayHandle) -> Self {
            Self {
                buffer_id: self.buffer_id,
                releases: self.releases.clone(),
            }
        }

        fn merge_into(self, into: &mut Self, _dh: &DisplayHandle) {
            if self.buffer_id.is_some() {
                *into = self;
            }
        }

        fn discard_with_retained(self, current: &mut Self, retained: &[&Self]) {
            let Some(buffer_id) = self.buffer_id else {
                return;
            };
            let buffer_is_retained = current.buffer_id == Some(buffer_id)
                || retained.iter().any(|state| state.buffer_id == Some(buffer_id));
            if !buffer_is_retained {
                self.releases
                    .expect("release probe with a buffer should have a release counter")
                    .fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    #[test]
    fn discard_state_range_exposes_retained_queued_states() {
        let _guard = DISCARD_TEST_LOCK.lock().unwrap();
        DISCARD_OBSERVATIONS.lock().unwrap().clear();
        let cache = Mutex::new(CachedState {
            pending: DiscardProbe::default(),
            current: DiscardProbe { id: 0 },
            cache: VecDeque::from([
                (Serial::from(1), DiscardProbe { id: 1 }),
                (Serial::from(2), DiscardProbe { id: 2 }),
                (Serial::from(3), DiscardProbe { id: 1 }),
            ]),
        });

        Cache::discard_state_range(&cache, Serial::from(1), Serial::from(1));

        assert_eq!(&*DISCARD_OBSERVATIONS.lock().unwrap(), &[(1, vec![2, 1])]);
    }

    #[test]
    fn discard_all_states_exposes_later_retained_states() {
        let _guard = DISCARD_TEST_LOCK.lock().unwrap();
        DISCARD_OBSERVATIONS.lock().unwrap().clear();
        let cache = Mutex::new(CachedState {
            pending: DiscardProbe::default(),
            current: DiscardProbe { id: 0 },
            cache: VecDeque::from([
                (Serial::from(1), DiscardProbe { id: 1 }),
                (Serial::from(2), DiscardProbe { id: 2 }),
                (Serial::from(3), DiscardProbe { id: 1 }),
            ]),
        });

        Cache::discard_all_states(&cache);

        assert_eq!(
            &*DISCARD_OBSERVATIONS.lock().unwrap(),
            &[(1, vec![2, 1]), (2, vec![1]), (1, vec![])]
        );
    }

    #[test]
    fn discard_state_range_defers_release_until_last_queued_reference() {
        let releases = Arc::new(AtomicUsize::new(0));
        let cache = Mutex::new(CachedState {
            pending: ReleaseProbe::default(),
            current: ReleaseProbe::default(),
            cache: VecDeque::from([
                (
                    Serial::from(1),
                    ReleaseProbe {
                        buffer_id: Some(7),
                        releases: Some(releases.clone()),
                    },
                ),
                (
                    Serial::from(2),
                    ReleaseProbe {
                        buffer_id: Some(7),
                        releases: Some(releases.clone()),
                    },
                ),
            ]),
        });

        Cache::discard_state_range(&cache, Serial::from(1), Serial::from(1));
        assert_eq!(
            releases.load(Ordering::SeqCst),
            0,
            "discarding one queued state must not release a buffer retained by another queued state"
        );

        Cache::discard_state_range(&cache, Serial::from(2), Serial::from(2));
        assert_eq!(
            releases.load(Ordering::SeqCst),
            1,
            "discarding the last queued reference should release the buffer exactly once"
        );
    }

    #[test]
    fn discard_state_range_does_not_release_current_buffer() {
        let releases = Arc::new(AtomicUsize::new(0));
        let cache = Mutex::new(CachedState {
            pending: ReleaseProbe::default(),
            current: ReleaseProbe {
                buffer_id: Some(7),
                releases: Some(releases.clone()),
            },
            cache: VecDeque::from([(
                Serial::from(1),
                ReleaseProbe {
                    buffer_id: Some(7),
                    releases: Some(releases.clone()),
                },
            )]),
        });

        Cache::discard_state_range(&cache, Serial::from(1), Serial::from(1));

        assert_eq!(
            releases.load(Ordering::SeqCst),
            0,
            "discarding queued state must not release a buffer retained by current state"
        );
    }

    #[test]
    fn discard_all_states_releases_after_last_queued_reference() {
        let releases = Arc::new(AtomicUsize::new(0));
        let cache = Mutex::new(CachedState {
            pending: ReleaseProbe::default(),
            current: ReleaseProbe::default(),
            cache: VecDeque::from([
                (
                    Serial::from(1),
                    ReleaseProbe {
                        buffer_id: Some(7),
                        releases: Some(releases.clone()),
                    },
                ),
                (
                    Serial::from(2),
                    ReleaseProbe {
                        buffer_id: Some(7),
                        releases: Some(releases.clone()),
                    },
                ),
            ]),
        });

        Cache::discard_all_states(&cache);

        assert_eq!(
            releases.load(Ordering::SeqCst),
            1,
            "discarding all queued references should release a retained buffer exactly once"
        );
    }
}
