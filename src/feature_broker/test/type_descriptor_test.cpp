#include <inference/type_descriptor.hpp>
#include "gtest/gtest.h"

using namespace ::inference;

TEST(InferenceTestSuite, TypeDescriptorCreation) {
    std::error_code code;
    auto descriptorInt = TypeDescriptor::CreateExpected<int>();
    ASSERT_TRUE(descriptorInt && !descriptorInt.value().IsUndefined());
}

TEST(InferenceTestSuite, TypeDescriptorEqualityChecks) {
    std::error_code code;
    auto descriptorInt = TypeDescriptor::CreateExpected<int>();
    ASSERT_TRUE(descriptorInt && !descriptorInt.value().IsUndefined());
    auto descriptorInt2 = TypeDescriptor::Create<int>();
    auto descriptorInt3 = TypeDescriptor::Create<int>();
    auto descriptorFloat = TypeDescriptor::Create<float>();

    ASSERT_EQ(descriptorInt.value(), descriptorInt2);
    ASSERT_EQ(descriptorInt.value(), descriptorInt3);
    ASSERT_EQ(descriptorInt2, descriptorInt3);
    ASSERT_NE(descriptorInt.value(), descriptorFloat);
}

namespace {
class YoDawgIHeardYouLikeClasses {};
class SoIPutAClassInYourClass {};
}  // namespace

TEST(InferenceTestSuite, TypeDescriptorNonBuiltInTypesBehavior) {
    std::error_code code;
    auto descriptor = TypeDescriptor::CreateExpected<::YoDawgIHeardYouLikeClasses>();

    if (TypeDescriptor::RuntimeTypesSupported)
    {
        // In this case it should be OK.
        ASSERT_TRUE(descriptor && !descriptor.value().IsUndefined());

        auto descriptor2 = TypeDescriptor::CreateExpected<::YoDawgIHeardYouLikeClasses>();
        ASSERT_TRUE(descriptor2 && !descriptor2.value().IsUndefined());
        ASSERT_EQ(descriptor.value(), descriptor2.value());

        auto descriptor3 = TypeDescriptor::CreateExpected<::SoIPutAClassInYourClass>();
        ASSERT_TRUE(descriptor3 && !descriptor3.value().IsUndefined());
        ASSERT_NE(descriptor.value(), descriptor3.value());
    }
    else
    {
        // In this case without RTTI the set of supported types is strongly constrained, so this should have failed.
        ASSERT_FALSE(descriptor);
        ASSERT_EQ(feature_errc::type_unsupported, descriptor.error());
    }
}
