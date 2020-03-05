# Remote Inferencing with Azure Personalizer Service
This provides a working example of a remote model where the inference step (UpdateIfChanged) will make a Http REST request to an Azure Personalizer Service for a recommendation. In order to build and use this example, the following needs to be done:
1) Install the CppRest SDK using vcpkg.
2) Create an Azure Personalizer instance through the Microsoft Azure portal. 
3) Enter the Uri and Subscription key into vw_remote_model/test/model_test.cpp file, denoted with ``ENTER_URI`` and ``ENTER_KEY``.
4) Uncomment the  ``add_subdirectory(vw_remote_model)`` in the CMakeLists.txt of the parent directory. 
5) Run cmake to generate the project

## Install CppRest using vcpkg
This assumes that you have vcpkg already installed. If you do not have vcpkg installed, please visit: https://github.com/microsoft/vcpkg for more information. Once vcpkg is installed, run the following command:
```
vcpkg.exe install cpprestsdk[core,compression]:x64-windows-static
```

This will provide a path to vcpkg's cmake file. Please copy this path as it is needed when running cmake for project setup. 

## Create an Azure Personalizer Service
In order to run this example, an instance of the Azure Personalizer Service must be created. The Azure Personalizer can be found on the [Azure portal](https://ms.portal.azure.com). To create the personalizer service, search for the resource "Personalizer". A resource group and storage account must also be created for the process. 

On a resource group of your choice create a new Personalizer. Enter a name for the personalizer, select a region to deploy it to, and click Create. 
![Create](img/createAPS1.jpg)

Once the deployment is complete, click "Go to resource", to see the newly deployed Personalizer. 

### Update APS policy

Update the Personalizer's policy, to include interactions (making the personalizer aware, that the context is the combination of the features). Please follow the highlights in the screenshots below: 1. Download the current policy 2. Update the policy file locally to be "--cb_explore_adf --epsilon 0.2 --power_t 0 -l 0.01 --cb_type mtr --interactions FFFi" 3. Upload the policy back to the personalizer, and wait a few seconds for the upload.
![Update](img/createAPS8.jpg) ![Update2](img/createAPS6.jpg)
![Update](img/createAPS7.jpg)

### Endpoint and Subscription Key

After the service is created, you will need to capture the endpoint and subscription key. These are located under **Keys and Endpoint**. This information will be needed by the code.

## Enter Uri and Subscription Key
The test file has two constants that contain the APS endpoint and the subscription key. This file will need to be updated with the information from creating the APS service. Therefore, edit FeatureBroker/src/vw_remote_model/test/model_test.cpp, replacing the ``ENTER_URI`` and ``ENTER_KEY`` with the values received from the Azure portal:
```
static std::string apsBaseUri = "ENTER_URI";
static std::string apsSubscriptionKey = "ENTER_KEY";
```

## Uncomment add_subdirectory for vw_remote_model
Because building this project is not part of the normal build, you will need to change the root level CMakeLists.txt file to reference the vw_remote_model directory. This can be done be editing FeatureBroker/CMakeLists.txt and changing:
```#add_subdirectory(vw_remote_model)```

to the following:
```add_subdirectory(vw_remote_model)```

## Running cmake
After completing the above steps, you can now run cmake to generate the project settings. To reference cpprest correctly, the CMake toolchain path must be updated to vcpkg's cmake file. This path was given after installing cpprest sdk. In addition, the x64 target for cpprest must also be specified  using the ``VCPKG_TARGET``. From the FeatureBroker/ do the following:
1) Make a build directory
 ```
 mkdir .\build
 cd .\build
 ```
2) Run cmake
```
cmake .. -DVCPKG_TARGET_TRIPLET=x64-windows-static  -DCMAKE_TOOL_CHAIN_FILE=<path to vcpkg/scripts/buildsystems/vcpkg.cmake> -G "Visual Studio 16 2019"
```