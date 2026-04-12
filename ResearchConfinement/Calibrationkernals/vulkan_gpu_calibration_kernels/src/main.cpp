#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulse_types.hpp"

namespace {

struct Args {
    std::string shaderDir = "shaders/bin";
    uint32_t taskMode = 0;
    uint32_t elementCount = 1024;
    uint32_t iterations = 8;
    float dt = 1.0f;
    float phaseScale = 1.0f;

    float frequencyNorm = 0.2450f;
    float amplitudeNorm = 0.18f;
    float voltageNorm = 0.36f;
    float currentNorm = 0.36f;
    float phaseRad = 0.25f;

    float sentSignal = 0.12f;
    float measuredSignal = 0.15f;
    float integratedFeedback = 0.30f;
    float derivativeSignal = 0.01f;

    float latticeClosure = 0.88f;
    float phaseClosure = 0.81f;
    float recurrenceAlignment = 0.76f;
    float conservationAlignment = 0.999f;
    float thermalNoise = 0.08f;
    float fieldNoise = 0.06f;
};

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

std::string escapeJson(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (char ch : text) {
        switch (ch) {
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            out += ch;
            break;
        }
    }
    return out;
}

float clamp01(float value) {
    return std::max(0.0f, std::min(1.0f, value));
}

std::string joinPath(const std::string& lhs, const std::string& rhs) {
    if (lhs.empty()) {
        return rhs;
    }
    const char tail = lhs.back();
    if (tail == '\\' || tail == '/') {
        return lhs + rhs;
    }
    return lhs + "/" + rhs;
}

std::vector<char> readFile(const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    const std::streamsize size = file.tellg();
    if (size <= 0) {
        throw std::runtime_error("Shader file is empty: " + path);
    }
    std::vector<char> buffer(static_cast<size_t>(size));
    file.seekg(0);
    file.read(buffer.data(), size);
    return buffer;
}

template <typename T>
T parseScalar(const std::string& text);

template <>
uint32_t parseScalar<uint32_t>(const std::string& text) {
    return static_cast<uint32_t>(std::stoul(text));
}

template <>
float parseScalar<float>(const std::string& text) {
    return std::stof(text);
}

Args parseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        const auto requireValue = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + name);
            }
            ++i;
            return std::string(argv[i]);
        };
        if (key == "--shader-dir") {
            args.shaderDir = requireValue("--shader-dir");
        } else if (key == "--task-mode") {
            args.taskMode = parseScalar<uint32_t>(requireValue("--task-mode"));
        } else if (key == "--element-count") {
            args.elementCount = std::max(1u, parseScalar<uint32_t>(requireValue("--element-count")));
        } else if (key == "--iterations") {
            args.iterations = std::max(1u, parseScalar<uint32_t>(requireValue("--iterations")));
        } else if (key == "--dt") {
            args.dt = parseScalar<float>(requireValue("--dt"));
        } else if (key == "--phase-scale") {
            args.phaseScale = parseScalar<float>(requireValue("--phase-scale"));
        } else if (key == "--frequency") {
            args.frequencyNorm = parseScalar<float>(requireValue("--frequency"));
        } else if (key == "--amplitude") {
            args.amplitudeNorm = parseScalar<float>(requireValue("--amplitude"));
        } else if (key == "--voltage") {
            args.voltageNorm = parseScalar<float>(requireValue("--voltage"));
        } else if (key == "--current") {
            args.currentNorm = parseScalar<float>(requireValue("--current"));
        } else if (key == "--phase-rad") {
            args.phaseRad = parseScalar<float>(requireValue("--phase-rad"));
        } else if (key == "--sent-signal") {
            args.sentSignal = parseScalar<float>(requireValue("--sent-signal"));
        } else if (key == "--measured-signal") {
            args.measuredSignal = parseScalar<float>(requireValue("--measured-signal"));
        } else if (key == "--integrated-feedback") {
            args.integratedFeedback = parseScalar<float>(requireValue("--integrated-feedback"));
        } else if (key == "--derivative-signal") {
            args.derivativeSignal = parseScalar<float>(requireValue("--derivative-signal"));
        } else if (key == "--lattice-closure") {
            args.latticeClosure = parseScalar<float>(requireValue("--lattice-closure"));
        } else if (key == "--phase-closure") {
            args.phaseClosure = parseScalar<float>(requireValue("--phase-closure"));
        } else if (key == "--recurrence-alignment") {
            args.recurrenceAlignment = parseScalar<float>(requireValue("--recurrence-alignment"));
        } else if (key == "--conservation-alignment") {
            args.conservationAlignment = parseScalar<float>(requireValue("--conservation-alignment"));
        } else if (key == "--thermal-noise") {
            args.thermalNoise = parseScalar<float>(requireValue("--thermal-noise"));
        } else if (key == "--field-noise") {
            args.fieldNoise = parseScalar<float>(requireValue("--field-noise"));
        } else {
            throw std::runtime_error("Unknown argument: " + key);
        }
    }
    return args;
}

VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &module) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module");
    }
    return module;
}

uint32_t findMemoryType(
    VkPhysicalDevice physicalDevice,
    uint32_t typeFilter,
    VkMemoryPropertyFlags properties
) {
    VkPhysicalDeviceMemoryProperties memProperties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
        const bool filterPass = (typeFilter & (1u << i)) != 0;
        const bool propertyPass = (memProperties.memoryTypes[i].propertyFlags & properties) == properties;
        if (filterPass && propertyPass) {
            return i;
        }
    }
    throw std::runtime_error("No compatible memory type found");
}

Buffer createBuffer(
    VkPhysicalDevice physicalDevice,
    VkDevice device,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties
) {
    Buffer out{};
    out.size = size;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bufferInfo, nullptr, &out.buffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer");
    }

    VkMemoryRequirements memRequirements{};
    vkGetBufferMemoryRequirements(device, out.buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &out.memory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate buffer memory");
    }
    if (vkBindBufferMemory(device, out.buffer, out.memory, 0) != VK_SUCCESS) {
        throw std::runtime_error("Failed to bind buffer memory");
    }
    return out;
}

template <typename T>
void uploadVector(VkDevice device, const Buffer& buffer, const std::vector<T>& data) {
    void* mapped = nullptr;
    if (vkMapMemory(device, buffer.memory, 0, buffer.size, 0, &mapped) != VK_SUCCESS) {
        throw std::runtime_error("Failed to map buffer for upload");
    }
    std::memcpy(mapped, data.data(), static_cast<size_t>(std::min<VkDeviceSize>(buffer.size, sizeof(T) * data.size())));
    vkUnmapMemory(device, buffer.memory);
}

template <typename T>
std::vector<T> downloadVector(VkDevice device, const Buffer& buffer, size_t count) {
    void* mapped = nullptr;
    if (vkMapMemory(device, buffer.memory, 0, buffer.size, 0, &mapped) != VK_SUCCESS) {
        throw std::runtime_error("Failed to map buffer for download");
    }
    std::vector<T> out(count);
    std::memcpy(out.data(), mapped, sizeof(T) * count);
    vkUnmapMemory(device, buffer.memory);
    return out;
}

void destroyBuffer(VkDevice device, Buffer& buffer) {
    if (buffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, buffer.buffer, nullptr);
        buffer.buffer = VK_NULL_HANDLE;
    }
    if (buffer.memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, buffer.memory, nullptr);
        buffer.memory = VK_NULL_HANDLE;
    }
    buffer.size = 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parseArgs(argc, argv);

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "vulkan_gpu_calibration";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "QuantumMiner";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_1;

        VkInstanceCreateInfo instanceInfo{};
        instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instanceInfo.pApplicationInfo = &appInfo;

        VkInstance instance = VK_NULL_HANDLE;
        if (vkCreateInstance(&instanceInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan instance");
        }

        uint32_t physicalDeviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr);
        if (physicalDeviceCount == 0) {
            throw std::runtime_error("No Vulkan physical devices found");
        }
        std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
        vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, physicalDevices.data());

        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        uint32_t queueFamilyIndex = std::numeric_limits<uint32_t>::max();
        VkPhysicalDeviceProperties physicalProps{};
        int bestScore = std::numeric_limits<int>::min();
        for (VkPhysicalDevice candidate : physicalDevices) {
            uint32_t queueFamilyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queueFamilyCount, nullptr);
            std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queueFamilyCount, queueFamilies.data());
            VkPhysicalDeviceProperties candidateProps{};
            vkGetPhysicalDeviceProperties(candidate, &candidateProps);
            int queueIndex = -1;
            for (uint32_t i = 0; i < queueFamilyCount; ++i) {
                if ((queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                    queueIndex = static_cast<int>(i);
                    break;
                }
            }
            if (queueIndex < 0) {
                continue;
            }
            int score = 0;
            if (candidateProps.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                score += 1000;
            } else if (candidateProps.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
                score += 100;
            }
            if (candidateProps.vendorID == 0x10DEu) {
                score += 250;
            }
            score += static_cast<int>(candidateProps.limits.maxComputeSharedMemorySize / 1024u);
            if (score > bestScore) {
                bestScore = score;
                physicalDevice = candidate;
                queueFamilyIndex = static_cast<uint32_t>(queueIndex);
                physicalProps = candidateProps;
            }
        }
        if (physicalDevice == VK_NULL_HANDLE || queueFamilyIndex == std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("No Vulkan compute queue family found");
        }

        const float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueInfo{};
        queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInfo.queueFamilyIndex = queueFamilyIndex;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &queuePriority;

        VkDeviceCreateInfo deviceInfo{};
        deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceInfo.queueCreateInfoCount = 1;
        deviceInfo.pQueueCreateInfos = &queueInfo;

        VkDevice device = VK_NULL_HANDLE;
        if (vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan device");
        }

        VkQueue queue = VK_NULL_HANDLE;
        vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

        const std::string calibrationPath = joinPath(args.shaderDir, "gpu_calibration.spv");
        const std::string trajectoryPath = joinPath(args.shaderDir, "trajectory_update.spv");
        const auto calibrationCode = readFile(calibrationPath);
        const auto trajectoryCode = readFile(trajectoryPath);

        VkShaderModule calibrationModule = createShaderModule(device, calibrationCode);
        VkShaderModule trajectoryModule = createShaderModule(device, trajectoryCode);

        std::vector<VkDescriptorSetLayoutBinding> bindings;
        for (uint32_t i = 0; i < 5; ++i) {
            VkDescriptorSetLayoutBinding binding{};
            binding.binding = i;
            binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binding.descriptorCount = 1;
            binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings.push_back(binding);
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor set layout");
        }

        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(vkcal::DispatchConfig);

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create pipeline layout");
        }

        auto createPipeline = [&](VkShaderModule module) -> VkPipeline {
            VkPipelineShaderStageCreateInfo stageInfo{};
            stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stageInfo.module = module;
            stageInfo.pName = "main";

            VkComputePipelineCreateInfo pipelineInfo{};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipelineInfo.stage = stageInfo;
            pipelineInfo.layout = pipelineLayout;

            VkPipeline pipeline = VK_NULL_HANDLE;
            if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create compute pipeline");
            }
            return pipeline;
        };

        VkPipeline calibrationPipeline = createPipeline(calibrationModule);
        VkPipeline trajectoryPipeline = createPipeline(trajectoryModule);

        const size_t elementCount = static_cast<size_t>(args.elementCount);
        std::vector<vkcal::PulseInput> pulseInputs(elementCount);
        std::vector<vkcal::FeedbackInput> feedbackInputs(elementCount);
        std::vector<vkcal::EnvironmentInput> environmentInputs(elementCount);
        std::vector<vkcal::CalibrationOutput> calibrationOutputs(elementCount);
        std::vector<vkcal::TrajectoryState> trajectoryStates(elementCount);

        for (size_t i = 0; i < elementCount; ++i) {
            const float phaseOffset = args.phaseRad + 0.03125f * static_cast<float>(i % 32);
            const float laneDrift = static_cast<float>(i % 17) / 512.0f;
            pulseInputs[i] = {
                clamp01(args.frequencyNorm + laneDrift * 0.25f),
                clamp01(args.amplitudeNorm + laneDrift * 0.10f),
                clamp01(args.voltageNorm + laneDrift * 0.08f),
                clamp01(args.currentNorm + laneDrift * 0.12f),
                phaseOffset,
                0.0f,
                0.0f,
                0.0f,
            };
            feedbackInputs[i] = {
                args.sentSignal,
                args.measuredSignal + laneDrift * 0.05f,
                args.integratedFeedback,
                args.derivativeSignal,
            };
            environmentInputs[i] = {
                clamp01(args.latticeClosure - laneDrift * 0.05f),
                clamp01(args.phaseClosure - laneDrift * 0.03f),
                clamp01(args.recurrenceAlignment - laneDrift * 0.04f),
                clamp01(args.conservationAlignment),
                clamp01(args.thermalNoise + laneDrift * 0.04f),
                clamp01(args.fieldNoise + laneDrift * 0.02f),
                0.0f,
                0.0f,
            };
            calibrationOutputs[i] = {};
            trajectoryStates[i].position = {0.0f, 0.0f, 0.0f, phaseOffset};
            trajectoryStates[i].velocity = {0.0010f, 0.0004f, -0.0002f, 0.0f};
        }

        Buffer pulseBuffer = createBuffer(
            physicalDevice,
            device,
            sizeof(vkcal::PulseInput) * elementCount,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );
        Buffer feedbackBuffer = createBuffer(
            physicalDevice,
            device,
            sizeof(vkcal::FeedbackInput) * elementCount,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );
        Buffer environmentBuffer = createBuffer(
            physicalDevice,
            device,
            sizeof(vkcal::EnvironmentInput) * elementCount,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );
        Buffer calibrationBuffer = createBuffer(
            physicalDevice,
            device,
            sizeof(vkcal::CalibrationOutput) * elementCount,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );
        Buffer trajectoryBuffer = createBuffer(
            physicalDevice,
            device,
            sizeof(vkcal::TrajectoryState) * elementCount,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );

        uploadVector(device, pulseBuffer, pulseInputs);
        uploadVector(device, feedbackBuffer, feedbackInputs);
        uploadVector(device, environmentBuffer, environmentInputs);
        uploadVector(device, calibrationBuffer, calibrationOutputs);
        uploadVector(device, trajectoryBuffer, trajectoryStates);

        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 5;

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;

        VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor pool");
        }

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &descriptorSetLayout;

        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate descriptor set");
        }

        std::array<VkDescriptorBufferInfo, 5> bufferInfos = {{
            {pulseBuffer.buffer, 0, pulseBuffer.size},
            {feedbackBuffer.buffer, 0, feedbackBuffer.size},
            {environmentBuffer.buffer, 0, environmentBuffer.size},
            {calibrationBuffer.buffer, 0, calibrationBuffer.size},
            {trajectoryBuffer.buffer, 0, trajectoryBuffer.size},
        }};

        std::array<VkWriteDescriptorSet, 5> descriptorWrites{};
        for (uint32_t i = 0; i < descriptorWrites.size(); ++i) {
            descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[i].dstSet = descriptorSet;
            descriptorWrites[i].dstBinding = i;
            descriptorWrites[i].dstArrayElement = 0;
            descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[i].descriptorCount = 1;
            descriptorWrites[i].pBufferInfo = &bufferInfos[i];
        }
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);

        VkCommandPoolCreateInfo commandPoolInfo{};
        commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolInfo.queueFamilyIndex = queueFamilyIndex;
        commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        VkCommandPool commandPool = VK_NULL_HANDLE;
        if (vkCreateCommandPool(device, &commandPoolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create command pool");
        }

        VkCommandBufferAllocateInfo commandAllocInfo{};
        commandAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandAllocInfo.commandPool = commandPool;
        commandAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandAllocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        if (vkAllocateCommandBuffers(device, &commandAllocInfo, &commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate command buffer");
        }

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("Failed to begin command buffer");
        }

        const uint32_t dispatchCount = std::max(1u, (args.elementCount + 63u) / 64u);
        vkcal::DispatchConfig config{};
        config.taskMode = args.taskMode;
        config.elementCount = args.elementCount;
        config.dt = args.dt;
        config.phaseScale = args.phaseScale;

        for (uint32_t iteration = 0; iteration < args.iterations; ++iteration) {
            vkCmdBindDescriptorSets(
                commandBuffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                pipelineLayout,
                0,
                1,
                &descriptorSet,
                0,
                nullptr
            );

            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, calibrationPipeline);
            vkCmdPushConstants(
                commandBuffer,
                pipelineLayout,
                VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                sizeof(vkcal::DispatchConfig),
                &config
            );
            vkCmdDispatch(commandBuffer, dispatchCount, 1, 1);

            VkMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(
                commandBuffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                1,
                &barrier,
                0,
                nullptr,
                0,
                nullptr
            );

            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, trajectoryPipeline);
            vkCmdPushConstants(
                commandBuffer,
                pipelineLayout,
                VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                sizeof(vkcal::DispatchConfig),
                &config
            );
            vkCmdDispatch(commandBuffer, dispatchCount, 1, 1);

            if (iteration + 1u < args.iterations) {
                vkCmdPipelineBarrier(
                    commandBuffer,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,
                    1,
                    &barrier,
                    0,
                    nullptr,
                    0,
                    nullptr
                );
            }
        }

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to end command buffer");
        }

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        const auto dispatchStarted = std::chrono::high_resolution_clock::now();
        if (vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit compute work");
        }
        vkQueueWaitIdle(queue);
        const auto dispatchFinished = std::chrono::high_resolution_clock::now();

        calibrationOutputs = downloadVector<vkcal::CalibrationOutput>(device, calibrationBuffer, elementCount);
        trajectoryStates = downloadVector<vkcal::TrajectoryState>(device, trajectoryBuffer, elementCount);

        double meanActuationGain = 0.0;
        double meanPulseSignal = 0.0;
        double meanPersistence = 0.0;
        double meanLeakage = 0.0;
        double meanPositionRadius = 0.0;
        double peakVelocity = 0.0;
        for (size_t i = 0; i < elementCount; ++i) {
            const auto& cal = calibrationOutputs[i];
            const auto& state = trajectoryStates[i];
            meanActuationGain += cal.actuationGain;
            meanPulseSignal += cal.pulseSignal;
            meanPersistence += cal.persistence;
            meanLeakage += cal.leakage;
            const double radius = std::sqrt(
                static_cast<double>(state.position.x * state.position.x)
                + static_cast<double>(state.position.y * state.position.y)
                + static_cast<double>(state.position.z * state.position.z)
            );
            meanPositionRadius += radius;
            const double velocity = std::sqrt(
                static_cast<double>(state.velocity.x * state.velocity.x)
                + static_cast<double>(state.velocity.y * state.velocity.y)
                + static_cast<double>(state.velocity.z * state.velocity.z)
            );
            peakVelocity = std::max(peakVelocity, velocity);
        }
        const double norm = static_cast<double>(elementCount);
        meanActuationGain /= norm;
        meanPulseSignal /= norm;
        meanPersistence /= norm;
        meanLeakage /= norm;
        meanPositionRadius /= norm;

        const auto clampSigned = [](double value) -> double {
            return std::max(-1.0, std::min(1.0, value));
        };
        const double frequencyNorm = clamp01(args.frequencyNorm);
        const double amplitudeNorm = clamp01(args.amplitudeNorm);
        const double resonanceGate = clamp01(static_cast<float>(0.5 * meanActuationGain + 0.5 * args.recurrenceAlignment));
        const double temporalOverlap = clamp01(static_cast<float>(meanPersistence));
        const double fluxGate = clamp01(static_cast<float>(meanLeakage + 0.5 * std::abs(meanPulseSignal)));
        const double energyGate = clamp01(static_cast<float>((0.50 * amplitudeNorm) + (0.25 * frequencyNorm) + (0.25 * clamp01(static_cast<float>(peakVelocity)))));
        const double jointGate = clamp01(static_cast<float>(std::sqrt(std::max(0.0, frequencyNorm * amplitudeNorm))));
        const double axisScaleX = clamp01(static_cast<float>(0.20 + 0.60 * frequencyNorm + 0.10 * resonanceGate + 0.10 * std::abs(meanActuationGain - meanLeakage)));
        const double axisScaleY = clamp01(static_cast<float>(0.20 + 0.60 * amplitudeNorm + 0.10 * temporalOverlap + 0.10 * std::abs(meanPersistence - meanPositionRadius)));
        const double axisScaleZ = clamp01(static_cast<float>(0.20 + 0.52 * jointGate + 0.16 * fluxGate + 0.12 * std::abs(meanPulseSignal)));
        const double axisResonance = clamp01(static_cast<float>(
            1.0 - (
                std::abs(axisScaleX - axisScaleY)
                + std::abs(axisScaleY - axisScaleZ)
                + std::abs(axisScaleX - axisScaleZ)
            ) / 3.0
        ));
        const double scaledX = (meanActuationGain - meanLeakage) * (0.5 + 0.5 * axisScaleX);
        const double scaledY = (meanPersistence - meanPositionRadius) * (0.5 + 0.5 * axisScaleY);
        const double scaledZ = meanPulseSignal * (0.5 + 0.5 * axisScaleZ);
        const double vectorEnergy = clamp01(static_cast<float>(
            std::sqrt(std::max(0.0, (scaledX * scaledX) + (scaledY * scaledY) + (scaledZ * scaledZ))) / std::sqrt(3.0)
            + 0.20 * energyGate
        ));
        const double speedMeasure = clamp01(static_cast<float>(0.52 * vectorEnergy + 0.26 * frequencyNorm + 0.22 * amplitudeNorm));
        const double gamma = 1.0 / std::sqrt(std::max(0.08, 1.0 - (0.92 * speedMeasure * speedMeasure)));
        const double relativisticCorrelation = clamp01(static_cast<float>((gamma - 1.0) / 2.5));
        const double temporalCouplingMoment = clamp01(static_cast<float>(
            0.34 * resonanceGate
            + 0.24 * axisResonance
            + 0.18 * temporalOverlap
            + 0.12 * jointGate
            + 0.12 * fluxGate
        ));
        const double spinAxisX = clampSigned((scaledY * axisScaleZ) - (scaledZ * axisScaleY));
        const double spinAxisY = clampSigned((scaledZ * axisScaleX) - (scaledX * axisScaleZ));
        const double spinAxisZ = clampSigned((scaledX * axisScaleY) - (scaledY * axisScaleX));
        const double spinMomentumScore = clamp01(static_cast<float>(
            std::sqrt(std::max(0.0, (spinAxisX * spinAxisX) + (spinAxisY * spinAxisY) + (spinAxisZ * spinAxisZ))) / std::sqrt(3.0)
        ));
        const double inertialMassProxy = clamp01(static_cast<float>(
            0.46 * vectorEnergy
            + 0.22 * relativisticCorrelation
            + 0.18 * spinMomentumScore
            + 0.14 * temporalCouplingMoment
        ));

        const auto elapsedNs = std::chrono::duration_cast<std::chrono::nanoseconds>(dispatchFinished - dispatchStarted).count();
        const double dispatchMs = static_cast<double>(elapsedNs) / 1000000.0;

        std::ostringstream json;
        json << std::fixed << std::setprecision(9);
        json << "{";
        json << "\"ok\":true";
        json << ",\"device_name\":\"" << escapeJson(physicalProps.deviceName) << "\"";
        json << ",\"vendor_id\":" << physicalProps.vendorID;
        json << ",\"device_id\":" << physicalProps.deviceID;
        json << ",\"task_mode\":" << args.taskMode;
        json << ",\"element_count\":" << args.elementCount;
        json << ",\"iterations\":" << args.iterations;
        json << ",\"dispatch_count\":" << dispatchCount;
        json << ",\"dispatch_elapsed_ms\":" << dispatchMs;
        json << ",\"mean_actuation_gain\":" << meanActuationGain;
        json << ",\"mean_pulse_signal\":" << meanPulseSignal;
        json << ",\"mean_persistence\":" << meanPersistence;
        json << ",\"mean_leakage\":" << meanLeakage;
        json << ",\"mean_position_radius\":" << meanPositionRadius;
        json << ",\"peak_velocity\":" << peakVelocity;
        json << ",\"axis_scale_x\":" << axisScaleX;
        json << ",\"axis_scale_y\":" << axisScaleY;
        json << ",\"axis_scale_z\":" << axisScaleZ;
        json << ",\"vector_energy\":" << vectorEnergy;
        json << ",\"temporal_coupling_moment\":" << temporalCouplingMoment;
        json << ",\"inertial_mass_proxy\":" << inertialMassProxy;
        json << ",\"relativistic_correlation\":" << relativisticCorrelation;
        json << ",\"spin_axis_x\":" << spinAxisX;
        json << ",\"spin_axis_y\":" << spinAxisY;
        json << ",\"spin_axis_z\":" << spinAxisZ;
        json << ",\"spin_momentum_score\":" << spinMomentumScore;
        json << ",\"phase_scale\":" << args.phaseScale;
        json << ",\"dt\":" << args.dt;
        json << "}";
        std::cout << json.str() << std::endl;

        vkDeviceWaitIdle(device);
        vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        destroyBuffer(device, trajectoryBuffer);
        destroyBuffer(device, calibrationBuffer);
        destroyBuffer(device, environmentBuffer);
        destroyBuffer(device, feedbackBuffer);
        destroyBuffer(device, pulseBuffer);
        vkDestroyPipeline(device, trajectoryPipeline, nullptr);
        vkDestroyPipeline(device, calibrationPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyShaderModule(device, trajectoryModule, nullptr);
        vkDestroyShaderModule(device, calibrationModule, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "{\"ok\":false,\"error\":\"" << escapeJson(exc.what()) << "\"}" << std::endl;
        return 1;
    }
}
