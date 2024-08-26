###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import logging

import pyhlml
import pynvml

from habana_frameworks.torch.gpu_migration.core.register import \
    BaseModuleRegister

logger = logging.getLogger(__name__)


class NVMLModuleRegister(BaseModuleRegister):
    @classmethod
    def get_module(cls):
        return pynvml


@NVMLModuleRegister.register_f()
def nvmlInit():
    """
    .. py:gpumgrcall:: nvmlInit.hpu_match

    Maps pynvml.nvmlInit to pyhlml.hlmlInit.

    """
    return pyhlml.hlmlInit()


@NVMLModuleRegister.register_f()
def nvmlShutdown():
    """
    .. py:gpumgrcall:: nvmlShutdown.hpu_match

    Maps pynvml.nvmlShutdown to pyhlml.hlmlShutdown.

    """
    return pyhlml.hlmlShutdown()


@NVMLModuleRegister.register_f()
def nvmlDeviceGetCount():
    """
    .. py:gpumgrcall:: nvmlDeviceGetCount.hpu_match

    Maps pynvml.nvmlDeviceGetCount to pyhlml.hlmlDeviceGetCount.

    """
    return pyhlml.hlmlDeviceGetCount()


@NVMLModuleRegister.register_f()
def nvmlDeviceGetHandleByIndex(index):
    """
    .. py:gpumgrcall:: nvmlDeviceGetHandleByIndex.hpu_match

    Maps pynvml.nvmlDeviceGetHandleByIndex to pyhlml.hlmlDeviceGetHandleByIndex.

    """
    return pyhlml.hlmlDeviceGetHandleByIndex(index)


@NVMLModuleRegister.register_f()
def nvmlDeviceGetCpuAffinity(handle, cpuSetSize):
    """
    .. py:gpumgrcall:: nvmlDeviceGetCpuAffinity.hpu_match

    Maps pynvml.nvmlDeviceGetCpuAffinity to pyhlml.hlmlDeviceGetCpuAffinity.

    """
    return pyhlml.hlmlDeviceGetCpuAffinity(handle, cpuSetSize)


@NVMLModuleRegister.register_f()
def nvmlDeviceClearCpuAffinity(handle):
    """
    .. py:gpumgrcall:: nvmlDeviceClearCpuAffinity.hpu_match

    Maps pynvml.nvmlDeviceClearCpuAffinity to pyhlml.hlmlDeviceClearCpuAffinity.

    """
    return pyhlml.hlmlDeviceClearCpuAffinity(handle)


@NVMLModuleRegister.register_f()
def nvmlDeviceGetClockInfo(handle, type):
    """
    .. py:gpumgrcall:: nvmlDeviceGetClockInfo.hpu_match

    Maps pynvml.nvmlDeviceGetClockInfo to pyhlml.hlmlDeviceGetClockInfo.

    """
    return pyhlml.hlmlDeviceGetClockInfo(handle, type)


@NVMLModuleRegister.register_f()
def nvmlDeviceGetCurrentClocksThrottleReasons(handle):
    """
    .. py:gpumgrcall:: nvmlDeviceGetCurrentClocksThrottleReasons.hpu_match

    Maps pynvml.nvmlDeviceGetCurrentClocksThrottleReasons to pyhlml.hlmlDeviceGetCurrentClocksThrottleReasons.

    """
    return pyhlml.hlmlDeviceGetCurrentClocksThrottleReasons(handle)


@NVMLModuleRegister.register_f()
def nvmlDeviceGetHandleByUUID(uuid):
    """
    .. py:gpumgrcall:: nvmlDeviceGetHandleByUUID.hpu_match

    Maps pynvml.nvmlDeviceGetHandleByUUID to pyhlml.hlmlDeviceGetHandleByUUID.

    """
    return pyhlml.hlmlDeviceGetHandleByUUID(uuid)


@NVMLModuleRegister.register_f()
def nvmlDeviceGetMaxClockInfo(handle, type):
    """
    .. py:gpumgrcall:: nvmlDeviceGetMaxClockInfo.hpu_match

    Maps pynvml.nvmlDeviceGetMaxClockInfo to pyhlml.hlmlDeviceGetMaxClockInfo.

    """
    return pyhlml.hlmlDeviceGetMaxClockInfo(handle, type)


@NVMLModuleRegister.register_f()
def nvmlDeviceGetMemoryErrorCounter(handle, errorType, counterType, locationType):
    """
    .. py:gpumgrcall:: nvmlDeviceGetMemoryErrorCounter.hpu_match

    Maps pynvml.nvmlDeviceGetMemoryErrorCounter to pyhlml.hlmlDeviceGetMemoryErrorCounter.

    """
    return pyhlml.hlmlDeviceGetMemoryErrorCounter(
        handle, errorType, counterType, locationType
    )


@NVMLModuleRegister.register_f()
def nvmlDeviceGetMemoryInfo(handle):
    """
    .. py:gpumgrcall:: nvmlDeviceGetMemoryInfo.hpu_match

    Maps pynvml.nvmlDeviceGetMemoryInfo to pyhlml.hlmlDeviceGetMemoryInfo.

    """
    return pyhlml.hlmlDeviceGetMemoryInfo(handle)


@NVMLModuleRegister.register_f()
def nvmlDeviceGetMinorNumber(handle):
    """
    .. py:gpumgrcall:: nvmlDeviceGetMinorNumber.hpu_match

    Maps pynvml.nvmlDeviceGetMinorNumber to pyhlml.hlmlDeviceGetMinorNumber.

    """
    return pyhlml.hlmlDeviceGetMinorNumber(handle)


@NVMLModuleRegister.register_f()
def nvmlDeviceGetName(handle):
    """
    .. py:gpumgrcall:: nvmlDeviceGetName.hpu_match

    Maps pynvml.nvmlDeviceGetName to pyhlml.hlmlDeviceGetName.

    """
    return pyhlml.hlmlDeviceGetName(handle)


@NVMLModuleRegister.register_f()
def nvmlDeviceGetPerformanceState(handle):
    """
    .. py:gpumgrcall:: nvmlDeviceGetPerformanceState.hpu_match

    Maps pynvml.nvmlDeviceGetPerformanceState to pyhlml.hlmlDeviceGetPerformanceState.

    """
    return pyhlml.hlmlDeviceGetPerformanceState(handle)


@NVMLModuleRegister.register_f()
def nvmlDeviceGetPersistenceMode(handle):
    """
    .. py:gpumgrcall:: nvmlDeviceGetPersistenceMode.hpu_match

    Maps pynvml.nvmlDeviceGetPersistenceMode to pyhlml.hlmlDeviceGetPersistenceMode.

    """
    return pyhlml.hlmlDeviceGetPersistenceMode()


@NVMLModuleRegister.register_f()
def nvmlDeviceGetPowerManagementDefaultLimit(handle):
    """
    .. py:gpumgrcall:: nvmlDeviceGetPowerManagementDefaultLimit.hpu_match

    Maps pynvml.nvmlDeviceGetPowerManagementDefaultLimit to pyhlml.hlmlDeviceGetPowerManagementDefaultLimit.

    """
    return pyhlml.hlmlDeviceGetPowerManagementDefaultLimit(handle)


@NVMLModuleRegister.register_f()
def nvmlDeviceGetPowerUsage(handle):
    """
    .. py:gpumgrcall:: nvmlDeviceGetPowerUsage.hpu_match

    Maps pynvml.nvmlDeviceGetPowerUsage to pyhlml.hlmlDeviceGetPowerUsage.

    """
    return pyhlml.hlmlDeviceGetPowerUsage(handle)


@NVMLModuleRegister.register_f()
def nvmlDeviceGetSerial(handle):
    """
    .. py:gpumgrcall:: nvmlDeviceGetSerial.hpu_match

    Maps pynvml.nvmlDeviceGetSerial to pyhlml.hlmlDeviceGetSerial.

    """
    return pyhlml.hlmlDeviceGetSerial(handle)


@NVMLModuleRegister.register_f()
def nvmlDeviceGetTemperature(handle, sensor):
    """
    .. py:gpumgrcall:: nvmlDeviceGetTemperature.hpu_match

    Maps pynvml.nvmlDeviceGetTemperature to pyhlml.hlmlDeviceGetTemperature.

    """
    return pyhlml.hlmlDeviceGetTemperature(handle, sensor)


@NVMLModuleRegister.register_f()
def nvmlDeviceGetTemperatureThreshold(handle, threshold):
    """
    .. py:gpumgrcall:: nvmlDeviceGetTemperatureThreshold.hpu_match

    Maps pynvml.nvmlDeviceGetTemperatureThreshold to pyhlml.hlmlDeviceGetTemperatureThreshold.

    """
    return pyhlml.hlmlDeviceGetTemperatureThreshold()


@NVMLModuleRegister.register_f()
def nvmlDeviceGetTotalEnergyConsumption(handle):
    """
    .. py:gpumgrcall:: nvmlDeviceGetTotalEnergyConsumption.hpu_match

    Maps pynvml.nvmlDeviceGetTotalEnergyConsumption to pyhlml.hlmlDeviceGetTotalEnergyConsumption.

    """
    return pyhlml.hlmlDeviceGetTotalEnergyConsumption(handle)


@NVMLModuleRegister.register_f()
def nvmlDeviceGetUUID(handle):
    """
    .. py:gpumgrcall:: nvmlDeviceGetUUID.hpu_match

    Maps pynvml.nvmlDeviceGetUUID to pyhlml.hlmlDeviceGetUUID.

    """
    return pyhlml.hlmlDeviceGetUUID(handle)


@NVMLModuleRegister.register_f()
def nvmlDeviceGetUtilizationRates(handle):
    """
    .. py:gpumgrcall:: nvmlDeviceGetUtilizationRates.hpu_match

    Maps pynvml.nvmlDeviceGetUtilizationRates to pyhlml.hlmlDeviceGetUtilizationRates.

    """
    return pyhlml.hlmlDeviceGetUtilizationRates(handle)


@NVMLModuleRegister.register_f()
def nvmlDeviceGetViolationStatus(device, perfPolicyType):
    """
    .. py:gpumgrcall:: nvmlDeviceGetViolationStatus.hpu_match

    Maps pynvml.nvmlDeviceGetViolationStatus to pyhlml.hlmlDeviceGetViolationStatus.

    """
    return pyhlml.hlmlDeviceGetViolationStatus(device, perfPolicyType)


@NVMLModuleRegister.register_f()
def nvmlDeviceRegisterEvents(handle, eventTypes, eventSet):
    """
    .. py:gpumgrcall:: nvmlDeviceRegisterEvents.hpu_match

    Maps pynvml.nvmlDeviceRegisterEvents to pyhlml.hlmlDeviceRegisterEvents.

    """
    return pyhlml.hlmlDeviceRegisterEvents(handle, eventTypes, eventSet)


@NVMLModuleRegister.register_f()
def nvmlDeviceSetCpuAffinity(handle):
    """
    .. py:gpumgrcall:: nvmlDeviceSetCpuAffinity.hpu_match

    Maps pynvml.nvmlDeviceSetCpuAffinity to pyhlml.hlmlDeviceSetCpuAffinity.

    """
    return pyhlml.hlmlDeviceSetCpuAffinity(handle)


@NVMLModuleRegister.register_f()
def nvmlEventSetCreate():
    """
    .. py:gpumgrcall:: nvmlEventSetCreate.hpu_match

    Maps pynvml.nvmlEventSetCreate to pyhlml.hlmlEventSetCreate.

    """
    return pyhlml.hlmlEventSetCreate()


@NVMLModuleRegister.register_f()
def nvmlEventSetFree(eventSet):
    """
    .. py:gpumgrcall:: nvmlEventSetFree.hpu_match

    Maps pynvml.nvmlEventSetFree to pyhlml.hlmlEventSetFree.

    """
    return pyhlml.hlmlEventSetFree(eventSet)


@NVMLModuleRegister.register_f()
def nvmlEventSetWait(eventSet, timeoutms):
    """
    .. py:gpumgrcall:: nvmlEventSetWait.hpu_match

    Maps pynvml.nvmlEventSetWait to pyhlml.hlmlEventSetWait.

    """
    return pyhlml.hlmlEventSetWait(eventSet, timeoutms)
