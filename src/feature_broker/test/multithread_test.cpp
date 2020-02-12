// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <condition_variable>
#include <inference/feature_broker.hpp>
#include <inference/feature_provider.hpp>
#include <inference/synchronous_feature_broker.hpp>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>

#include "add_five_model.hpp"
#include "add_model.hpp"
#include "gtest/gtest.h"
#include "release_model.hpp"
#include "tuple_feature_providers.hpp"

using namespace ::inference;

template <typename T>
class FBSyncAsyncTest : public ::testing::Test {
   public:
    using FeatureBrokerType = T;
};

// There appears to be a problem with gtest type parameterized tests on G++ specifically. This:
//     fb->BindInput<int>(stream.str());
// Gives this error on G++ on linux.
//     expected primary-expression before ‘int’".
// But this works just fine:
//     BindInputHelp<int>(fb, stream.str());
// Calls on non-templated methods work fine. Calling the templated methods from anywhere other than a TYPED_TEST works
// fine (as we see below). Calling templated members on anything other than the member type derived from this also works
// fine. I have no clear idea why this would be problematic.

template <typename T>
rt::expected<std::shared_ptr<DirectInputPipe<T>>> BindInputHelp(std::shared_ptr<FeatureBroker> const& fb,
                                                                std::string const& name) {
    return fb->BindInput<T>(name);
}

template <typename T>
rt::expected<std::shared_ptr<DirectInputPipe<int>>> BindInputHelp(std::shared_ptr<SynchronousFeatureBroker> const& fb,
                                                                  std::string const& name) {
    return fb->BindInput<T>(name);
}

using FbTypes = ::testing::Types<::inference::FeatureBroker, ::inference::SynchronousFeatureBroker>;
TYPED_TEST_CASE(FBSyncAsyncTest, FbTypes);

TYPED_TEST(FBSyncAsyncTest, MultiThreadInputPipe) {
    using TypeParam = typename TestFixture::FeatureBrokerType;
    // If the following were a `const` int, VSC++ would insist that it be captured to be used in the lambda while Clang
    // would insist that it *not* be captured since that capture is, by its standards, unnecessary. G++ works either
    // way.
    int lim = 1000;
    int count1 = 0, count2 = 0;
    auto model = std::make_shared<inference_test::AddFiveModel>();
    auto fb = std::make_shared<TypeParam>(model);
    auto func1 = [&fb, &count1, &lim]() -> void {
        std::stringstream stream;
        bool atLeastOneFailure = false;
        for (int i = 0; i < lim; ++i) {
            stream << i;
            // In an ideal world the following would not be needed...
            auto expectedInput = BindInputHelp<int>(fb, stream.str());
            if (!expectedInput) {
                ++count1;
                atLeastOneFailure = true;
            } else if (atLeastOneFailure)  // Once we fail to add, we should continue to.
                return;
            stream.str("");
        }
    };

    auto func2 = [&fb, &count2, &lim]() -> void {
        std::stringstream stream;
        bool atLeastOneFailure = false;
        for (int i = lim; i > 0;) {
            stream << --i;
            auto expectedInput = BindInputHelp<int>(fb, stream.str());
            if (!expectedInput) {
                ++count2;
                atLeastOneFailure = true;
            } else if (atLeastOneFailure)
                return;
            stream.str("");
        }
    };

    std::thread thread1(func1);
    func2();
    thread1.join();
    ASSERT_EQ(lim, count1 + count2);
}

TYPED_TEST(FBSyncAsyncTest, MultiThreadInputProvider) {
    using TypeParam = typename TestFixture::FeatureBrokerType;
    int lim = 100;
    int count1 = 0, count2 = 0;
    auto model = std::make_shared<inference_test::AddFiveModel>();
    auto fb = std::make_shared<TypeParam>(model);
    auto func1 = [&fb, &count1, &lim]() -> void {
        std::stringstream stream;
        bool atLeastOneFailure = false;
        for (int i = 0; i < lim; ++i) {
            stream << i;
            std::string s = stream.str();
            stream.str("");

            auto provider = inference_test::TupleProviderFactory::Create<int, int>("A" + s, "B" + s);
            auto expectedInput = fb->BindInputs(provider);
            if (!expectedInput) {
                ASSERT_EQ(feature_errc::already_bound, expectedInput.error());
                ++count1;
                atLeastOneFailure = true;
            } else if (atLeastOneFailure)  // Once we fail to add, we should continue to.
                return;
        }
    };

    auto func2 = [&fb, &count2, &lim]() -> void {
        std::stringstream stream;
        bool atLeastOneFailure = false;
        for (int i = lim; i > 0;) {
            stream << --i;
            std::string s = stream.str();
            stream.str("");
            auto provider = inference_test::TupleProviderFactory::Create<int, int>("A" + s, "B" + s);

            auto expectedInput = fb->BindInputs(provider);
            if (!expectedInput) {
                ASSERT_EQ(feature_errc::already_bound, expectedInput.error());
                ++count2;
                atLeastOneFailure = true;
            } else if (atLeastOneFailure)
                return;
        }
    };

    std::thread thread1(func1);
    func2();
    thread1.join();
    ASSERT_EQ(lim, count1 + count2);
}

class NameCollisionTestProvider final : public FeatureProvider {
   public:
    explicit NameCollisionTestProvider(const bool first) {
        const auto type = TypeDescriptor::Create<int>();
        std::vector<std::string> names;
        // Notice how they collide at M.
        if (first)
            names = {"B", "C", "M", "W", "X"};
        else
            names = {"D", "E", "M", "Y", "Z"};
        for (const auto& name : names) _outputs.emplace(name, type);
    }

    std::unordered_map<std::string, TypeDescriptor> const& Outputs() const override { return _outputs; }

    rt::expected<std::shared_ptr<ValueUpdater>> CreateValueUpdater(
        std::map<std::string, std::shared_ptr<InputPipe>> const& outputToPipe,
        std::function<void()> valuesChangedNotifier) const override {
        return std::shared_ptr<ValueUpdater>(new UpdaterImpl());
    }

   private:
    class UpdaterImpl final : public ValueUpdater {
       public:
        std::error_code UpdateOutput() override { return {}; }
    };

    std::unordered_map<std::string, TypeDescriptor> _outputs;
};

// Somewhat like the .NET System.Threading.Barrier, simplified with just SignalAndWait, and some additional assumptions
// about usage.
class Barrier {
   public:
    explicit Barrier(int barrierSize) : _barrierSize(barrierSize), _waitingFor(barrierSize), _wakingUp(0) {}

    void SignalAndWait() {
        std::unique_lock<std::mutex> lock(mutex);
        if (--_waitingFor == 0) {
            // One lucky waiter gets to wake everyone up. We update the global state, so that the next cycle it knows to
            // update. Otherwise there are sort of "fake" wakeups. (Couldn't this be built into the CV?)
            // Reset the waiters.
            _waitingFor = _barrierSize;
            _wakingUp = _barrierSize - 1;
            _cv.notify_all();
        } else {
            // These waiters get to wait. Note that passing this in releases the lock.
            // Without the predicate and keeping track of the tick, sometimes things seem to wake up anyway??
            // If this was real we'd probably want to reset
            _cv.wait(lock, [this] { return _wakingUp > 0; });
            // Set it back to false. I think this should work, so long as it is:
            // 1. always the same group of threads waiting and
            // 2. the barriers come in groups of two or more.
            // Otherwise there are some race conditions. Of course these are really bad assumptions, but probably good
            // enough for test code.
            --_wakingUp;
        }
    }

   private:
    std::mutex mutex;
    std::condition_variable _cv;

    const int _barrierSize;
    int _waitingFor;
    int _wakingUp;
};

TYPED_TEST(FBSyncAsyncTest, MultiThreadInputProviderNoDoubling) {
    using TypeParam = typename TestFixture::FeatureBrokerType;
    int lim = 100;
    auto model = std::make_shared<inference_test::AddFiveModel>();
    std::shared_ptr<TypeParam> fb;
    bool success1, success2;

    // The way this works is that two worker threads both own a feature provider, that have different features, except
    // for an overlap of one. We start the

    Barrier startBarrier(3), endBarrier(3);
    auto worker = [&fb, &startBarrier, &endBarrier, &lim](bool providerArg, bool& success) -> void {
        auto provider = std::make_shared<NameCollisionTestProvider>(providerArg);
        for (int i = 0; i < lim; ++i) {
            startBarrier.SignalAndWait();
            auto expectedBind = fb->BindInputs(provider);
            // One should succeed, one should fail.
            success = (bool)expectedBind;
            endBarrier.SignalAndWait();
        }
    };

    std::thread thread1(worker, true, std::ref(success1));
    std::thread thread2(worker, false, std::ref(success2));

    int successCount1 = 0;
    const auto success1Names = {"B", "C", "W", "X"};
    const auto success2Names = {"D", "E", "Y", "Z"};

    // The main thread is the one that always sets things.
    for (int i = 0; i < lim; ++i) {
        // In our first exclusive section, we set a new feature broker.
        fb = std::make_shared<TypeParam>(model);
        startBarrier.SignalAndWait();  // Release the workers.
        endBarrier.SignalAndWait();    // Wait for the workers.
        // One of these should succeed, the other should fail.
        ASSERT_NE(success1, success2);
        if (success1) ++successCount1;  // Mostly for inspection during debugging.

        // Check the current binding status to make sure it's consistent...
        // In all cases,
        auto bindExpected = BindInputHelp<int>(fb, "M");
        ASSERT_FALSE(bindExpected);

        // If 1 succeeded, the names in success1Names should all be bound, and the names in success2Names unbound.
        if (success1) {
            {
                for (const auto& name : success1Names) {
                    bindExpected = BindInputHelp<int>(fb, name);
                    ASSERT_FALSE(bindExpected);
                    ASSERT_EQ(feature_errc::already_bound, bindExpected.error());
                }
                for (const auto& name : success2Names) {
                    // These bindings should succeed now, since they failed earlier.
                    bindExpected = BindInputHelp<int>(fb, name);
                    ASSERT_TRUE(bindExpected && bindExpected.value());
                }
            }
        } else {
            for (const auto& name : success1Names) {
                bindExpected = BindInputHelp<int>(fb, name);
                ASSERT_TRUE(bindExpected && bindExpected.value());
            }
            for (const auto& name : success2Names) {
                bindExpected = BindInputHelp<int>(fb, name);
                ASSERT_FALSE(bindExpected);
                ASSERT_EQ(feature_errc::already_bound, bindExpected.error());
            }
        }
    }
    thread1.join();
    thread2.join();
}

class SetParentTestProvider final : public FeatureProvider {
   public:
    explicit SetParentTestProvider(std::shared_ptr<Barrier> barrier)
        : _outputs({{"A", TypeDescriptor::Create<float>()}}), _barrier(barrier), _active(false) {}

    std::unordered_map<std::string, TypeDescriptor> const& Outputs() const override {
        // This will be queried as part of one of the calls to SetParent, to check for a type mismatch.
        std::unique_lock<std::mutex> lock(_mutex);
        if (_active) {
            _barrier->SignalAndWait();
            _active = false;
        }
        return _outputs;
    }

    rt::expected<std::shared_ptr<ValueUpdater>> CreateValueUpdater(
        std::map<std::string, std::shared_ptr<InputPipe>> const& outputToPipe,
        std::function<void()> valuesChangedNotifier) const override {
        return std::shared_ptr<ValueUpdater>(new UpdaterImpl());
    }

    void Activate() { _active = true; }

   private:
    class UpdaterImpl final : public ValueUpdater {
       public:
        std::error_code UpdateOutput() override { return {}; }
    };

    std::unordered_map<std::string, TypeDescriptor> _outputs;
    std::shared_ptr<Barrier> _barrier;
    mutable bool _active;
    mutable std::mutex _mutex;
};

TEST(InferenceTestSuite, MultiThreadFeatureBrokerSetParent) {
    // To test the thread safety of SetParent, we try to introduce a cycle. However, this must be done somewhat
    // carefully, for a few reasons:

    // First, even without synchronization, it is likely that even without proper synchronization that it may succeed,
    // so we have to wait till a very specific point till the second thread begins its work to make sure it's the
    // "worst" possible time.

    // Second, the point of the cycle detection is to avoid infinite loops, so we are introducing an operation that may
    // conceivably induce an infinite loop, so we have to be very careful about how this test is structured.

    int lim = 100;
    // The purpose of the model is to make sure that the FeatureBroker asks about the input "A", so that the Barrier we
    // are trying to trip gets tripped.
    auto model = std::make_shared<inference_test::AddFiveModel>();

    // The purpose of this barrier is to feed it to the feature provider, so that at the "right" time,
    auto barrier = std::make_shared<Barrier>(2);
    auto fp = std::make_shared<SetParentTestProvider>(barrier);

    for (int i = 0; i < lim; ++i) {
        auto fb1 = std::make_shared<FeatureBroker>();
        auto bindExpected = fb1->BindInputs(fp);
        ASSERT_TRUE((bool)bindExpected);
        auto fb2 = std::make_shared<FeatureBroker>(model);

        // Now, let's make some trouble. We first start so that our feature provider "tripwire" is activated...
        fp->Activate();
        // Next we set up the thread worker. This thread worker will try to make fb1 fb2's parent. This will block on
        // the barrier.

        rt::expected<void> threadExpected;

        auto worker = [&fb1, &fb2, &threadExpected]() -> void { threadExpected = fb2->SetParent(fb1); };
        std::thread thread(worker);
        barrier->SignalAndWait();
        // Past this point, we go in the opposite direction, making fb2 fb1's parent. However it should be too late,
        // since we only got to this point past when the feature provider released our barrier, at which point the
        // worker was already in the critical section of the worker thread.
        auto expected = fb1->SetParent(fb2);
        ASSERT_FALSE(expected);
        ASSERT_EQ(feature_errc::circular_structure, expected.error());
        thread.join();
    }
}

TEST(InferenceTestSuite, MultiThreadFeatureBrokerWaitUntilChanged) {
    auto model = std::make_shared<inference_test::AddFiveModel>();
    auto fb = std::make_shared<FeatureBroker>(model);
    auto input = fb->BindInput<float>("A").value_or(nullptr);
    auto output = fb->BindOutput<float>("X").value_or(nullptr);
    ASSERT_NE(nullptr, input);
    ASSERT_NE(nullptr, output);
    Barrier barrier(2);

    float value = 0;
    auto worker = [&value, &output, &barrier]() -> void {
        barrier.SignalAndWait();
        output->WaitUntilChanged();
        // This should only be cleared.
        ASSERT_EQ(1, value);
        ASSERT_TRUE(output->Changed());
        auto expectedUpdate = output->UpdateIfChanged(value);
        ASSERT_TRUE(expectedUpdate && expectedUpdate.value());
    };

    std::thread thread(worker);
    barrier.SignalAndWait();
    // Now both the thread and this thread are working. But, at this stage, the thread should always wait until changed.
    // Sleeping about 1 ms should give that thread a chance to at least reach the wait on the output pipe, but it should
    // go no further.
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    // The value should not have been changed yet, since that thread worker should have still done nothing but wait.
    ASSERT_EQ(0, value);
    // Set a value to compare against in the thread. It should be that once it wakes up.
    value = 1;
    // Set the value, hence waking up the thread.
    input->Feed(3);
    // Wait for the thread to finish.
    thread.join();
    // Check that the thread updated the value to the correct value as part of its work.
    ASSERT_EQ(8, value);
}

TEST(InferenceTestSuite, MultiThreadFeatureBrokerWaitUntilChangedComplex) {
    auto model = std::make_shared<inference_test::AddModel>();
    auto fb = std::make_shared<FeatureBroker>(model);
    auto inputA = fb->BindInput<float>("A").value_or(nullptr);
    auto inputB = fb->BindInput<float>("B").value_or(nullptr);
    auto output = fb->BindOutput<float>("X").value_or(nullptr);
    ASSERT_NE(nullptr, inputA);
    ASSERT_NE(nullptr, inputB);
    ASSERT_NE(nullptr, output);

    Barrier barrier(2), barrierEnd(2);
    bool newOutputExpected(false), done(false);
    float valueExpected;

    auto worker = [&output, &barrier, &barrierEnd, &newOutputExpected, &valueExpected, &done]() -> void {
        float value;
        while (!done) {
            barrier.SignalAndWait();
            if (newOutputExpected) {
                output->WaitUntilChanged();
                ASSERT_TRUE(output->Changed());
                auto outputExpected = output->UpdateIfChanged(value);
                ASSERT_TRUE(outputExpected && outputExpected.value());
                ASSERT_EQ(valueExpected, value);
                ASSERT_FALSE(output->Changed());
            } else {
                ASSERT_FALSE(output->Changed());
                auto outputExpected = output->UpdateIfChanged(value);
                ASSERT_TRUE(outputExpected && !outputExpected.value());
            }
            barrierEnd.SignalAndWait();
        }
    };

    std::thread thread(worker);

    barrier.SignalAndWait();
    barrierEnd.SignalAndWait();
    // Past this first round, should not have output.

    // Still shouldn't be ready...
    inputA->Feed(1);
    barrier.SignalAndWait();
    barrierEnd.SignalAndWait();

    // Should be ready now.
    inputB->Feed(2);
    valueExpected = 3;
    newOutputExpected = true;
    barrier.SignalAndWait();
    barrierEnd.SignalAndWait();

    // Just do a quick re-run, make sure it hasn't changed.
    newOutputExpected = false;
    barrier.SignalAndWait();
    barrierEnd.SignalAndWait();

    // Feed in a new input, make sure it changes.
    inputA->Feed(3);
    valueExpected = 5;
    newOutputExpected = true;
    barrier.SignalAndWait();
    barrierEnd.SignalAndWait();

    newOutputExpected = false;

    barrier.SignalAndWait();
    done = true;
    barrierEnd.SignalAndWait();
    thread.join();
}

TEST(InferenceTestSuite, MultiThreadFeatureBrokerWaitUntilChangedModelSideChannel) {
    auto internalModel = std::make_shared<inference_test::AddFiveModel>();
    auto model = std::make_shared<inference_test::ReleaseModel>(internalModel);
    auto fb = std::make_shared<FeatureBroker>(model);
    auto input = fb->BindInput<float>("A").value_or(nullptr);
    auto output = fb->BindOutput<float>("X").value_or(nullptr);
    ASSERT_NE(nullptr, input);
    ASSERT_NE(nullptr, output);
    Barrier barrier(2), barrierEnd(2);

    float value = 0;
    auto worker = [&value, &output, &barrier, &barrierEnd]() -> void {
        barrier.SignalAndWait();
        barrierEnd.SignalAndWait();
        // Checkpoint 1.
        ASSERT_FALSE(output->Changed());

        barrier.SignalAndWait();
        barrierEnd.SignalAndWait();
        // Checkpoint 2. The input has been fed, but the model is not released.
        ASSERT_FALSE(output->Changed());
        auto expectedUpdate = output->UpdateIfChanged(value);
        ASSERT_TRUE(expectedUpdate && !expectedUpdate.value());

        barrier.SignalAndWait();
        barrierEnd.SignalAndWait();
        // Checkpoint 3. The model is released, so we ought to be able to query it now.
        output->WaitUntilChanged();
        ASSERT_TRUE(output->Changed());
        expectedUpdate = output->UpdateIfChanged(value);
        ASSERT_TRUE(expectedUpdate && expectedUpdate.value());
        ASSERT_EQ(8, value);
        ASSERT_FALSE(output->Changed());
    };

    std::thread thread(worker);
    barrier.SignalAndWait();
    // Pre-checkpoint 1.
    barrierEnd.SignalAndWait();

    barrier.SignalAndWait();
    // Pre-checkpoint 2.
    input->Feed(3);
    barrierEnd.SignalAndWait();

    barrier.SignalAndWait();
    // Pre-checkpoint 3.
    model->Release();
    barrierEnd.SignalAndWait();

    thread.join();
}
