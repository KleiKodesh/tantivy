//! Background resource priority for the calling thread.
//!
//! Used by `IndexWriterOptions::low_priority_threads` for the indexing worker
//! threads and for the merges the merge policy starts on its own, and public
//! so that an application can run its own feeding thread the same way.
//!
//! A thread in background mode gets CPU, and where the OS supports it disk
//! and memory, only when nobody else asks for them:
//! - Windows: `SetThreadPriority(GetCurrentThread(), THREAD_MODE_BACKGROUND_BEGIN)` lowers the
//!   CPU, I/O and memory priority of the thread; fallback `THREAD_PRIORITY_LOWEST`.
//!   <https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setthreadpriority>
//! - Linux: `sched_setscheduler(0, SCHED_IDLE, ..)` (per thread; pid 0 = the calling thread) plus
//!   `ioprio_set(IOPRIO_WHO_PROCESS, 0, IOPRIO_CLASS_IDLE)` (who 0 = the calling thread).
//!   `SCHED_IDLE` only when the thread may leave it again (its nice value inside
//!   `RLIMIT_NICE`), else the always reversible `SCHED_BATCH`.
//!   <https://man7.org/linux/man-pages/man2/sched_setscheduler.2.html>,
//!   <https://man7.org/linux/man-pages/man7/sched.7.html>,
//!   <https://man7.org/linux/man-pages/man2/ioprio_set.2.html>
//! - Android: the calling thread's nice value is set to 10, Android's
//!   `THREAD_PRIORITY_BACKGROUND` (nice is per thread on Linux), plus the idle I/O class.
//!   <https://developer.android.com/reference/android/os/Process#THREAD_PRIORITY_BACKGROUND>,
//!   <https://man7.org/linux/man-pages/man2/setpriority.2.html>
//! - Apple platforms: `pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND, 0)` (lower CPU and
//!   I/O throughput).
//!   <https://developer.apple.com/library/archive/documentation/Performance/Conceptual/EnergyGuide-iOS/PrioritizeWorkWithQoS.html>
//! - Elsewhere: nothing.
//!
//! All of it is best effort: a refused call leaves the thread as it was, and
//! the work runs either way.
//!
//! Never hold background mode across a section that foreground work waits
//! for (a commit, a lock a reader needs): the OS boosts a starved thread's
//! CPU priority at most, never its I/O, so a waiter would stall for as long
//! as the disk is busy. Wrap such sections in [`NormalPrioritySection`].
//! A thread in background mode that WAITS for background work (e.g.
//! `IndexWriter::wait_merging_threads` on merges the policy started) waits
//! at that work's pace; wait from a normal-priority thread when it matters.

use std::cell::Cell;
use std::marker::PhantomData;

thread_local! {
    /// Nesting depth of background scopes on this thread: only 0 -> 1 enters
    /// the OS mode and only 1 -> 0 leaves it (the OS mode is one flag per
    /// thread, so a nested end would otherwise end the outer scope's mode).
    static DEPTH: Cell<u32> = const { Cell::new(0) };
    /// What entering the OS mode changed, to undo it; `None` while out of it.
    static SAVED: Cell<Option<imp::Saved>> = const { Cell::new(None) };
    /// Set while a [`NormalPrioritySection`] has left the OS mode.
    static SUSPENDED: Cell<bool> = const { Cell::new(false) };
}

/// Enters background mode on the calling thread (nesting-safe). Pair every
/// call with [`end_background`] on the same thread; prefer
/// [`BackgroundPriority`] in Rust, this pair is for callers that cannot hold
/// a guard (a C ABI).
pub fn begin_background() {
    let depth = DEPTH.with(|d| {
        let depth = d.get();
        d.set(depth + 1);
        depth
    });
    if depth == 0 && !SUSPENDED.with(Cell::get) {
        SAVED.with(|s| s.set(imp::begin()));
    }
}

/// Leaves what the matching [`begin_background`] entered. Unmatched calls
/// are ignored.
pub fn end_background() {
    let depth = DEPTH.with(|d| {
        let depth = d.get();
        d.set(depth.saturating_sub(1));
        depth
    });
    if depth == 1 {
        if let Some(saved) = SAVED.with(|s| s.take()) {
            imp::end(saved);
        }
    }
}

/// True while the calling thread is inside a background scope (and not in a
/// [`NormalPrioritySection`]).
pub fn is_background() -> bool {
    DEPTH.with(Cell::get) > 0 && !SUSPENDED.with(Cell::get)
}

/// Background mode for the calling thread while the guard lives. Nesting is
/// safe; the guard must be dropped on the thread that created it.
pub struct BackgroundPriority {
    _not_send: PhantomData<*const ()>,
}

impl BackgroundPriority {
    /// Enters background mode on the calling thread.
    pub fn enter() -> BackgroundPriority {
        begin_background();
        BackgroundPriority {
            _not_send: PhantomData,
        }
    }
}

impl Drop for BackgroundPriority {
    fn drop(&mut self) {
        end_background();
    }
}

/// Normal priority for the calling thread while the guard lives, if it was in
/// background mode (a no-op otherwise): for the short sections foreground
/// work waits for. Background mode comes back when the guard drops.
pub struct NormalPrioritySection {
    active: bool,
    _not_send: PhantomData<*const ()>,
}

impl NormalPrioritySection {
    /// Leaves background mode until the guard drops.
    pub fn enter() -> NormalPrioritySection {
        let active = is_background();
        if active {
            SUSPENDED.with(|s| s.set(true));
            if let Some(saved) = SAVED.with(|s| s.take()) {
                imp::end(saved);
            }
        }
        NormalPrioritySection {
            active,
            _not_send: PhantomData,
        }
    }
}

impl Drop for NormalPrioritySection {
    fn drop(&mut self) {
        if self.active {
            SUSPENDED.with(|s| s.set(false));
            if DEPTH.with(Cell::get) > 0 {
                SAVED.with(|s| s.set(imp::begin()));
            }
        }
    }
}

#[cfg(windows)]
mod imp {
    use winapi::um::processthreadsapi::{GetCurrentThread, GetThreadPriority, SetThreadPriority};
    use winapi::um::winbase::{
        THREAD_MODE_BACKGROUND_BEGIN, THREAD_MODE_BACKGROUND_END, THREAD_PRIORITY_ERROR_RETURN,
        THREAD_PRIORITY_LOWEST,
    };

    #[derive(Clone, Copy)]
    pub(super) enum Saved {
        /// `THREAD_MODE_BACKGROUND_BEGIN` succeeded.
        Mode,
        /// Fallback: the CPU priority before `THREAD_PRIORITY_LOWEST`.
        Priority(i32),
    }

    pub(super) fn begin() -> Option<Saved> {
        // SAFETY: `GetCurrentThread` is a pseudo handle for the calling
        // thread, the only handle the background mode values accept.
        unsafe {
            let thread = GetCurrentThread();
            if SetThreadPriority(thread, THREAD_MODE_BACKGROUND_BEGIN as i32) != 0 {
                return Some(Saved::Mode);
            }
            let old = GetThreadPriority(thread);
            if old == THREAD_PRIORITY_ERROR_RETURN as i32 {
                return None;
            }
            if SetThreadPriority(thread, THREAD_PRIORITY_LOWEST as i32) != 0 {
                return Some(Saved::Priority(old));
            }
            None
        }
    }

    pub(super) fn end(saved: Saved) {
        // SAFETY: as in `begin`.
        unsafe {
            let thread = GetCurrentThread();
            match saved {
                Saved::Mode => SetThreadPriority(thread, THREAD_MODE_BACKGROUND_END as i32),
                Saved::Priority(old) => SetThreadPriority(thread, old),
            };
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
mod imp {
    // <linux/ioprio.h>
    const IOPRIO_WHO_PROCESS: libc::c_int = 1;
    const IOPRIO_CLASS_SHIFT: libc::c_int = 13;
    const IOPRIO_CLASS_IDLE: libc::c_int = 3;

    #[derive(Clone, Copy)]
    pub(super) struct Saved {
        /// The I/O priority before the idle class, if that was set.
        ioprio: Option<libc::c_int>,
        /// The CPU side before: (policy, priority) for `SCHED_IDLE`, or the
        /// nice value (Android).
        cpu: Option<Cpu>,
    }

    #[derive(Clone, Copy)]
    enum Cpu {
        #[cfg_attr(target_os = "android", allow(dead_code))]
        Policy(libc::c_int, libc::c_int),
        #[cfg_attr(target_os = "linux", allow(dead_code))]
        Nice(libc::c_int),
    }

    fn ioprio_get() -> libc::c_int {
        // SAFETY: plain syscall on the calling thread (who 0).
        unsafe { libc::syscall(libc::SYS_ioprio_get, IOPRIO_WHO_PROCESS, 0) as libc::c_int }
    }

    fn ioprio_set(value: libc::c_int) -> bool {
        // SAFETY: plain syscall on the calling thread (who 0).
        unsafe { libc::syscall(libc::SYS_ioprio_set, IOPRIO_WHO_PROCESS, 0, value) == 0 }
    }

    #[cfg(target_os = "linux")]
    fn lower_cpu() -> Option<Cpu> {
        // SAFETY: pid 0 = the calling thread; `param` is a valid out pointer.
        unsafe {
            let policy = libc::sched_getscheduler(0);
            let mut param: libc::sched_param = std::mem::zeroed();
            if policy < 0 || libc::sched_getparam(0, &mut param) != 0 {
                return None;
            }
            // Leaving SCHED_IDLE needs the thread's nice value inside the
            // RLIMIT_NICE range (sched(7)); many systems default the limit to
            // 0, which forbids it. A mode that cannot be left would hold the
            // CPU side across the normal sections, so then SCHED_BATCH (the
            // non-interactive policy, always reversible) is used instead.
            let tid = libc::syscall(libc::SYS_gettid) as libc::id_t;
            let nice = libc::getpriority(libc::PRIO_PROCESS, tid);
            let mut limit: libc::rlimit = std::mem::zeroed();
            let reversible = libc::getrlimit(libc::RLIMIT_NICE, &mut limit) == 0
                && (limit.rlim_cur == libc::RLIM_INFINITY
                    || (20 - nice as i64) <= limit.rlim_cur as i64);
            let target = if reversible {
                libc::SCHED_IDLE
            } else {
                libc::SCHED_BATCH
            };
            let zero = libc::sched_param { sched_priority: 0 };
            (libc::sched_setscheduler(0, target, &zero) == 0)
                .then_some(Cpu::Policy(policy, param.sched_priority))
        }
    }

    #[cfg(target_os = "android")]
    fn lower_cpu() -> Option<Cpu> {
        const THREAD_PRIORITY_BACKGROUND: libc::c_int = 10;
        // SAFETY: plain syscalls on the calling thread's own id.
        unsafe {
            // getpriority on our own thread id cannot fail (-1 is then a
            // valid nice value)
            let tid = libc::syscall(libc::SYS_gettid) as libc::id_t;
            let old = libc::getpriority(libc::PRIO_PROCESS, tid);
            (libc::setpriority(libc::PRIO_PROCESS, tid, old.max(THREAD_PRIORITY_BACKGROUND)) == 0)
                .then_some(Cpu::Nice(old))
        }
    }

    fn restore_cpu(cpu: Cpu) {
        // SAFETY: plain syscalls on the calling thread. `lower_cpu` only
        // enters a policy this thread may leave; a refusal anyway leaves it
        // throttled, slower but never incorrect.
        unsafe {
            match cpu {
                Cpu::Policy(policy, priority) => {
                    let param = libc::sched_param {
                        sched_priority: priority,
                    };
                    libc::sched_setscheduler(0, policy, &param);
                }
                Cpu::Nice(nice) => {
                    let tid = libc::syscall(libc::SYS_gettid) as libc::id_t;
                    libc::setpriority(libc::PRIO_PROCESS, tid, nice);
                }
            }
        }
    }

    pub(super) fn begin() -> Option<Saved> {
        let old_io = ioprio_get();
        let ioprio =
            (old_io >= 0 && ioprio_set(IOPRIO_CLASS_IDLE << IOPRIO_CLASS_SHIFT)).then_some(old_io);
        let cpu = lower_cpu();
        (ioprio.is_some() || cpu.is_some()).then_some(Saved { ioprio, cpu })
    }

    pub(super) fn end(saved: Saved) {
        if let Some(cpu) = saved.cpu {
            restore_cpu(cpu);
        }
        if let Some(io) = saved.ioprio {
            ioprio_set(io);
        }
    }
}

#[cfg(target_vendor = "apple")]
mod imp {
    #[derive(Clone, Copy)]
    pub(super) struct Saved(libc::qos_class_t, libc::c_int);

    pub(super) fn begin() -> Option<Saved> {
        // SAFETY: only reads and changes the calling thread's QoS.
        unsafe {
            let mut class = libc::qos_class_t::QOS_CLASS_UNSPECIFIED;
            let mut relative = 0;
            if libc::pthread_get_qos_class_np(libc::pthread_self(), &mut class, &mut relative) != 0
            {
                return None;
            }
            (libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_BACKGROUND, 0) == 0)
                .then_some(Saved(class, relative))
        }
    }

    pub(super) fn end(saved: Saved) {
        let Saved(mut class, mut relative) = saved;
        if matches!(class, libc::qos_class_t::QOS_CLASS_UNSPECIFIED) {
            // a thread without a QoS class: back to the default one
            class = libc::qos_class_t::QOS_CLASS_DEFAULT;
            relative = 0;
        }
        // SAFETY: only changes the calling thread's QoS.
        unsafe {
            libc::pthread_set_qos_class_self_np(class, relative);
        }
    }
}

#[cfg(not(any(
    windows,
    target_os = "linux",
    target_os = "android",
    target_vendor = "apple"
)))]
mod imp {
    #[derive(Clone, Copy)]
    pub(super) struct Saved;

    pub(super) fn begin() -> Option<Saved> {
        None
    }

    pub(super) fn end(_: Saved) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    fn on_new_thread(f: impl FnOnce() + Send + 'static) {
        std::thread::spawn(f).join().unwrap();
    }

    #[test]
    fn scopes_nest_and_unwind() {
        on_new_thread(|| {
            assert!(!is_background());
            {
                let _outer = BackgroundPriority::enter();
                assert!(is_background());
                {
                    let _inner = BackgroundPriority::enter();
                    assert!(is_background());
                }
                // the inner scope must not end the outer one
                assert!(is_background());
                {
                    let _normal = NormalPrioritySection::enter();
                    assert!(!is_background());
                    // a scope entered inside a normal section stays normal
                    let _again = BackgroundPriority::enter();
                    assert!(!is_background());
                }
                assert!(is_background());
            }
            assert!(!is_background());
            // a normal section outside any scope is a no-op
            let _normal = NormalPrioritySection::enter();
            assert!(!is_background());
        });
    }

    #[test]
    fn unmatched_end_is_ignored() {
        on_new_thread(|| {
            end_background();
            begin_background();
            assert!(is_background());
            end_background();
            assert!(!is_background());
        });
    }

    #[cfg(windows)]
    #[test]
    fn windows_background_mode_is_entered_and_left() {
        use winapi::um::processthreadsapi::{GetCurrentThread, SetThreadPriority};
        use winapi::um::winbase::THREAD_MODE_BACKGROUND_END;
        on_new_thread(|| {
            let scope = BackgroundPriority::enter();
            // In the mode, END succeeds once; put the mode back for the scope.
            let in_mode = unsafe {
                SetThreadPriority(GetCurrentThread(), THREAD_MODE_BACKGROUND_END as i32) != 0
            };
            assert!(in_mode, "background mode was not entered");
            SAVED.with(|s| s.set(imp::begin()));
            drop(scope);
            // Left: END fails ("not in background processing mode").
            let still = unsafe {
                SetThreadPriority(GetCurrentThread(), THREAD_MODE_BACKGROUND_END as i32) != 0
            };
            assert!(!still, "background mode was not left");
        });
    }
}
