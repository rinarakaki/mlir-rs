/*!
# Async Runtime Reference Implementation

This file declares basic Async runtime API for supporting Async dialect to LLVM dialect lowering.

- include <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/ExecutionEngine/AsyncRuntime.h>
- lib <https://github.com/llvm/llvm-project/blob/main/mlir/lib/ExecutionEngine/AsyncRuntime.cpp>
*/

// Adds references to reference counted runtime object.
pub extern "C" fn mlir_async_runtime_add_ref(ptr: RefCountedObjPtr, count: i64) {
    let ref_counted = static_cast::<*mut RefCounted>(ptr);
    ref_counted.add_ref(count);
}

// Drops references from reference counted runtime object.
extern "C" fn mlir_async_runtime_drop_ref(ptr: RefCountedObjPtr, count: i64) {
    let ref_counted = static_cast::<*mut RefCounted>(ptr);
    ref_counted.drop_ref(count);
}

// Create a new `async.token` in not-ready state.
extern "C" fn mlir_async_runtime_create_token() -> *mut AsyncToken {
    AsyncToken::new(default_async_runtime())
}

/**
Create a new `async.value` in not-ready state. Size parameter specifies the number of bytes that will be allocated for the async value storage. Storage is owned by the `async.value` and deallocated when the async value is destructed (reference count drops to zero).
*/
extern "C" fn mlir_async_runtime_create_value(size: i64) -> *mut AsyncValue {
    AsyncValue::new(default_async_runtime(), size)
}

// Create a new `async.group` in empty state.
extern "C" fn mlir_async_runtime_create_group(size: i64) -> *mut AsyncGroup {
    AsyncGroup::new(default_async_runtime(), size)
}

extern "C" fn mlir_async_runtime_add_token_to_group(
    token: *mut AsyncToken,
    group: *mut AsyncGroup)
-> i64
{
    std::unique_lock<std::mutex> lock_token(token.mu);
    std::unique_lock<std::mutex> lock_group(group.mu);

    // Get the rank of the token inside the group before we drop the reference.
    let rank = group.rank.fetch_add(1);

    let on_token_ready = || {
        // Increment the number of errors in the group.
        if State(token.state).is_error() {
            group.num_errors.fetch_add(1);
        }

        /*
        If pending tokens go below zero it means that more tokens than the group size were added to this group.
        */
        assert!(group.pending_tokens > 0, "Wrong group size");

        // Run all group awaiters if it was the last token in the group.
        if group.pending_tokens.fetch_sub(1) == 1 {
            group.cv.notify_all();
            for awaiter in group.awaiters {
                awaiter();
            }
        }
    };

    if State(token.state).is_available_or_error() {
        // Update group pending tokens immediately and maybe run awaiters.
        on_token_ready();

    } else {
        /*
        Update group pending tokens when token will become ready. Because this will happen asynchronously we must ensure that `group` is alive until then, and re-ackquire the lock.
        */
        group.add_ref();

        token.awaiters.emplace_back(|| {
            // Make sure that `drop_ref` does not destroy the mutex owned by the lock.
            {
                std::unique_lock<std::mutex> lock_group(group.mu);
                on_token_ready();
            }
            group.drop_ref();
        });
    }

    rank
}

/*
Switches `async.token` to available or error state (terminatl state) and runs all awaiters.
*/
pub fn set_token_state(token: *mut AsyncToken, state: State) {
    assert!(state.is_available_or_error(), "must be terminal state");
    assert!(State(token.state).is_unavailable(), "token must be unavailable");
  
    // Make sure that `drop_ref` does not destroy the mutex owned by the lock.
    {
        std::unique_lock<std::mutex> lock(token.mu);
        token.state = state;
        token.cv.notify_all();
        for awaiter in token.awaiters {
            awaiter();
        }
    }
  
    // Async tokens created with a ref count `2` to keep token alive until the
    // async task completes. Drop this reference explicitly when token emplaced.
    token.drop_ref();
}
  
pub fn set_value_state(value: *mut AsyncValue, state: State) {
    assert!(state.is_available_or_error(), "must be terminal state");
    assert!(State(value.state).is_unavailable(), "value must be unavailable");
  
    // Make sure that `drop_ref` does not destroy the mutex owned by the lock.
    {
        std::unique_lock<std::mutex> lock(value.mu);
        value.state = state;
        value.cv.notify_all();
        for awaiter in value.awaiters {
            awaiter();
        }
    }
  
    // Async values created with a ref count `2` to keep value alive until the
    // async task completes. Drop this reference explicitly when value emplaced.
    value.drop_ref();
}
  

// Switches `async.token` to ready state and runs all awaiters.
extern "C" fn mlir_async_runtime_emplace_token(token: *mut AsyncToken) {
    set_token_state(token, State::Available);
}  

// Switches `async.value` to ready state and runs all awaiters.
extern "C" fn mlir_async_runtime_emplace_value(value: *mut AsyncValue) {
    set_value_state(value, State::Available);
}

// Switches `async.token` to error state and runs all awaiters.
extern "C" fn mlir_async_runtime_set_token_error(token: *mut AsyncToken) {
    set_token_state(token, State::Error);
}

// Switches `async.value` to error state and runs all awaiters.
extern "C" fn mlir_async_runtime_set_value_error(value: *mut AsyncValue) {
    set_value_state(value, State::Error)
}

// Returns true if token is in the error state.
extern "C" fn mlir_async_runtime_is_token_error(token: *mut AsyncToken) -> bool {
    State(token.state).is_error()
}

/// Returns true if value is in the error state.
extern "C" fn mlir_async_runtime_is_value_error(value: *mut AsyncValue) -> bool {
    State(value.state).is_error()
}

/**
Returns true if group is in the error state (any of the tokens or values added to the group are in the error state).
*/
extern "C" fn mlir_async_runtime_is_group_error(group: *mut AsyncGroup) -> bool {
    group.num_errors.load() > 0
}

/// Blocks the caller thread until the token becomes ready.
extern "C" fn mlir_async_runtime_await_token(token: *mut AsyncToken) {
    std::unique_lock<std::mutex> lock(token.mu);
    if !State(token.state).is_available_or_error() {
        token.cv.wait(
            lock,
            || State(token.state).is_available_or_error());
    }
}

/// Blocks the caller thread until the value becomes ready.
extern "C" fn mlir_async_runtime_await_value(value: *mut AsyncValue) {
    std::unique_lock<std::mutex> lock(value.mu);
    if !State(value.state).is_available_or_error() {
        value.cv.wait(
            lock,
            || State(value.state).is_available_or_error());
    }
}

/// Blocks the caller thread until the elements in the group become ready.
extern "C" fn mlir_async_runtime_await_all_in_group(group: *mut AsyncGroup) {
    std::unique_lock<std::mutex> lock(group.mu);
    if group.pending_tokens != 0 {
        group.cv.wait(lock, || group.pending_tokens == 0);
    }
}

/// Returns a pointer to the storage owned by the async value.
extern "C" fn mlir_async_runtime_get_value_storage(value: *mut AsyncValue) -> ValueStorage {
    assert!(!State(value.state).is_error(), "Unexpected error state");
    value.storage.data()
}

/**
Executes the task (coro handle + resume function) in one of the threads managed by the runtime.
*/
extern "C" fn mlir_async_runtime_execute(handle: CoroHandle, resume: CoroResume) {
    let runtime = default_async_runtime();
    runtime.thread_pool().r#async(|| { (*resume)(handle); });
}

// Executes the task (coro handle + resume function) in one of the threads
// managed by the runtime after the token becomes ready.
extern "C" fn mlir_async_runtime_await_token_and_execute(
    token: *mut AsyncToken,
    handle: CoroHandle,
    resume: CoroResume)
{
    let execute = || { (*resume)(handle); };
    std::unique_lock<std::mutex> lock(token.mu);
    if State(token.state).is_available_or_error() {
        lock.unlock();
        execute();
    } else {
        token.awaiters.emplace_back(|| { execute(); });
    }
}

/**
Executes the task (coro handle + resume function) in one of the threads managed by the runtime after the value becomes ready.
*/
extern "C" fn mlir_async_runtime_await_value_and_execute(
    value: *mut AsyncValue,
    handle: CoroHandle,
    resume: CoroResume)
{
    let execute = || { (*resume)(handle); };
    std::unique_lock<std::mutex> lock(value.mu);
    if State(value.state).is_available_or_error() {
        lock.unlock();
        execute();
    } else {
        value.awaiters.emplace_back(|| { execute(); });
    }
}

/**
Executes the task (coro handle + resume function) in one of the threads managed by the runtime after the all members of the group become ready.
*/
extern "C" fn mlir_async_runtime_await_all_in_group_and_execute(
    group: *mut AsyncGroup,
    handle: CoroHandle,
    resume: CoroResume)
{
    let execute = || { (*resume)(handle); };
    std::unique_lock<std::mutex> lock(group.mu);
    if group.pending_tokens == 0 {
        lock.unlock();
        execute();
    } else {
        group.awaiters.emplace_back(|| { execute(); });
    }
}


// Returns the current number of available worker threads in the threadpool.
extern "C" fn mlir_async_runtim_get_num_worker_threads() -> i64 {
    static thread_local std::thread::id this_id = std::this_thread::get_id();
    std::cout << "Current thread id: " << this_id << std::endl;
}

//===----------------------------------------------------------------------===//
// Small async runtime support library for testing.
//===----------------------------------------------------------------------===//

extern "C" fn mlir_async_runtime_print_current_thread_id() {

}

// -------------------------------------------------------------------------- //
// AsyncRuntime orchestrates all async operations and Async runtime API is built
// on top of the default runtime instance.
// -------------------------------------------------------------------------- //

pub struct AsyncRuntime {
    num_ref_counted_objects: AtomicI64,
    thread_pool: llvm::ThreadPool
}

impl AsyncRuntime {
public:
    pub fn new() -> Self {
        Self {
            num_ref_counted_objects: 0,
            ..Default::default()
        }
    }

    ~AsyncRuntime() {
        thread_pool.wait(); // wait for the completion of all async tasks
        assert(self.num_ref_counted_objects() == 0 &&
            "all ref counted objects must be destroyed");
    }

    pub fn num_ref_counted_objects(&self) -> i64 {
        self.num_ref_counted_objects.load(std::memory_order_relaxed)
    }

    pub fn thread_pool(&self) -> &llvm::ThreadPool { return thread_pool; }

private:
    friend class RefCounted;

    /**
    Count the total number of reference counted objects in this instance of an AsyncRuntime. For debugging purposes only.
    */
    fn add_num_ref_counted_objects(&mut self) {
        self.num_ref_counted_objects.fetch_add(1, std::memory_order_relaxed);
    }
    fn drop_num_ref_counted_objects(&mut self) {
        self.num_ref_counted_objects.fetch_sub(1, std::memory_order_relaxed);
    }
}

// -------------------------------------------------------------------------- //
// A state of the async runtime value (token, value or group).
// -------------------------------------------------------------------------- //

pub struct State {
    state: StateKind
}

impl State {
public:
    /* implicit */ State(StateKind s) : state(s) {}
    /* implicit */ operator StateKind() { return state; }

    bool is_unavailable() const { return state == Unavailable; }
    bool is_available() const { return state == Available; }
    bool is_error() const { return state == Error; }
    bool is_available_or_error() const { return is_available() || is_error(); }

    const char *debug() const {
        match state {
            Unavailable =>
            return "unavailable",
            Available =>
            return "available",
            Error =>
            return "error"
        }
    }
}

#[repr(i8)]
pub enum StateKind {
    /// The underlying value is not yet available for consumption.
    Unavailable = 0,
    /**
    The underlying value is available for consumption. This state can not transition to any other state.
    */
    Available = 1,
    /**
    This underlying value is available and contains an error. This state can not transition to any other state.
    */
    Error = 2,
}

// -------------------------------------------------------------------------- //
// A base class for all reference counted objects created by the async runtime.
// -------------------------------------------------------------------------- //

pub struct RefCounted {
    runtime: *mut AsyncRuntime,
    ref_count: AtomicI64
}

impl RefCounted {
    pub fn new(runtime: *mut AsyncRuntime, ref_count: i64 /* 1 */) -> Self {
        runtime.add_num_ref_counted_objects();
        Self { runtime, ref_count }
    }

    pub fn add_ref(&self, count: i64 /* 1 */) {
        self.ref_count.fetch_add(count);
    }

    pub fn drop_ref(&self, count: i64 /* 1 */) {
        let previous = self.ref_count.fetch_sub(count);
        assert!(previous >= count, "Reference count should not go below zero");
        if previous == count {
            destroy();
        }
    }

    fn destroy(self) {}
}

impl Drop for RefCounted {
    fn drop(&mut self) {
        assert!(ref_count.load() == 0, "reference count must be zero");
        runtime.drop_num_ref_counted_objects();
    }
}

/// Returns the default per-process instance of an async runtime.
pub fn default_async_runtime_instance() -> &std::unique_ptr<AsyncRuntime> {
    static runtime = std::make_unique<AsyncRuntime>();
    return runtime;
}

pub fn reset_default_async_runtime() {
    default_async_runtime_instance().reset()
}

pub fn default_async_runtime() -> *mut AsyncRuntime {
   default_async_runtime_instance().get()
}

/// Async token provides a mechanism to signal asynchronous operation completion.
pub struct AsyncToken {
    state: std::atomic<State::StateKind>,

    // Pending awaiters are guarded by a mutex.
    mu: std::mutex,
    cv: std::condition_variable,
    awaiters: Vec<std::function<void()>>
}

impl AsyncToken : public RefCounted {
    /*
    AsyncToken created with a reference count of 2 because it will be returned
    to the `async.execute` caller and also will be later on emplaced by the
    asynchronously executed task. If the caller immediately will drop its
    reference we must ensure that the token will be alive until the
    asynchronous operation is completed.
    */
    AsyncToken(AsyncRuntime *runtime)
        : RefCounted(runtime, /*ref_count=*/2), state(State::Unavailable) {}
}

/**
Async value provides a mechanism to access the result of asynchronous
operations. It owns the storage that is used to store/load the value of the
underlying type, and a flag to signal if the value is ready or not.
*/
pub struct AsyncValue {
    state: std::atomic<State::StateKind>

    // Use vector of bytes to store async value payload.
    storage: Vec<u8>,
  
    // Pending awaiters are guarded by a mutex.
    mu: std::mutex,
    cv: std::condition_variable,
    awaiters: Vec<std::function<void()>>
}

impl AsyncValue : public RefCounted {
    /// AsyncValue similar to an AsyncToken created with a reference count of 2.
    pub fn new(AsyncRuntime *runtime, size: i64)
        : RefCounted(runtime, /*ref_count=*/2), state(State::Unavailable),
            storage(size) {}
}

/**
Async group provides a mechanism to group together multiple async tokens or values to await on all of them together (wait for the completion of all tokens or values added to the group).
*/
pub struct AsyncGroup {
    pending_tokens: std::atomic<int>,
    num_errors: std::atomic<int>,
    rank: std::atomic<int>,

    // Pending awaiters are guarded by a mutex.
    mu: std::mutex,
    cv: std::condition_variable,
    awaiters: Vec<std::function<void()>>
}

impl AsyncGroup : public RefCounted {
    pub fn new(AsyncRuntime *runtime, size: i64)
        : RefCounted(runtime), pending_tokens(size), num_errors(0), rank(0) {}
}
