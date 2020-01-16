#include <inference/feature_error.hpp>
#include <inference/input_pipe.hpp>

namespace inference {

InputPipe::OutputWaiter::OutputWaiter(size_t waiters) : _waiters(waiters), _ready(waiters == 0) {}

void InputPipe::OutputWaiter::Ping(bool subsequentCall) {
    std::unique_lock<std::mutex> lock(_mutex);
    if (subsequentCall) {
        if (_waiters == 0) {
            _ready = true;
            _cv.notify_one();
        }
        // If we still have some waiters, then this subsequent call should have no effect, yet.
    } else if (--_waiters == 0) {
        _ready = true;
        _cv.notify_one();
    }
}

std::error_code InputPipe::OutputWaiter::Wait() {
    std::unique_lock<std::mutex> lock(_mutex);
    if (!_ready) {
        // The thing that calls this method are the output pipe implementations, and part of the contract with the
        // API is that client code should treat the individual pipes themselves as being synchronous structures
        // (even though among themselves they enable asynchronous behavior).
        if (_waiting) return make_feature_error(feature_errc::multiple_waiting);
        _waiting = true;
        // No need to capture all of this, just the reference to the bool will suffice.
        bool& ready = _ready;
        _cv.wait(lock, [&ready]() { return ready; });
        _waiting = false;
    }
    _ready = false;
    return {};
}

bool InputPipe::OutputWaiter::Cleared() {
    std::unique_lock<std::mutex> lock(_mutex);
    return _waiters == 0;
}

}  // namespace inference
