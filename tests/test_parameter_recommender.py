from core.hardware import CPU, GPU, Hardware, Memory, System
from core.knowledge_base import KnowledgeBase
from core.model_info import ModelInfo
from core.parameter_recommender import ParameterRecommender


def create_hardware() -> Hardware:
    return Hardware(
        system=System(
            os="test",
            kernel="test",
            architecture="x86_64",
        ),
        cpu=CPU(
            vendor="Intel",
            model="Test CPU",
            physical_cores=24,
            logical_cores=24,
        ),
        memory=Memory(
            total_bytes=48 * 1024**3,
            available_bytes=40 * 1024**3,
            used_bytes=8 * 1024**3,
            swap_total_bytes=8 * 1024**3,
            swap_used_bytes=0,
        ),
        gpus=[
            GPU(
                vendor="NVIDIA",
                model="RTX 3060",
                memory_type="dedicated",
                dedicated=True,
                vram_total_bytes=12 * 1024**3,
                vram_used_bytes=0,
                vram_free_bytes=12 * 1024**3,
                pci_vendor_id="0x10de",
                pci_device_id="0x2504",
                compute_api=["CUDA"],
            )
        ],
    )


def create_recommender() -> ParameterRecommender:
    kb = KnowledgeBase("knowledge/hardware_profiles.json")
    kb.load()
    return ParameterRecommender(kb)


def test_parameter_profile_contains_core_parameters():
    model = ModelInfo(
        name="test",
        parameters_b=8.0,
        quantization="Q4_K_M",
        context_length=65536,
    )

    profile = create_recommender().recommend(
        create_hardware(),
        model,
        65536,
    )

    values = profile.as_dict()

    assert values["num_ctx"] == 65536
    assert values["num_gpu"] == -1
    assert values["num_thread"] == 24
    assert values["temperature"] == 0.7
    assert values["top_k"] == 40
    assert values["top_p"] == 0.9
    assert values["min_p"] == 0.05
    assert values["repeat_penalty"] == 1.1
    assert values["num_predict"] == 8192
    assert values["keep_alive"] == "10m"


def test_parameter_profile_has_explanations():
    model = ModelInfo(
        name="test",
        parameters_b=8.0,
        quantization="Q4_K_M",
        context_length=65536,
    )

    profile = create_recommender().recommend(
        create_hardware(),
        model,
        65536,
    )

    assert profile.parameters

    for parameter in profile.parameters:
        assert parameter.name
        assert parameter.explanation


def test_parameter_recommendation_uses_live_free_vram():
    hardware = create_hardware()

    hardware.gpus[0].vram_used_bytes = (
        10 * 1024**3
    )
    hardware.gpus[0].vram_free_bytes = (
        2 * 1024**3
    )

    model = ModelInfo(
        name="large-test",
        parameters_b=8.0,
        quantization="Q4_K_M",
        context_length=65536,
    )

    profile = create_recommender().recommend(
        hardware,
        model,
        65536,
    )

    values = profile.as_dict()

    # Con solo 2 GiB libres el modelo no debe tratarse
    # como si tuviera disponibles los 12 GiB completos.
    assert values["num_batch"] == 128


def test_parameter_profile_contains_capacity_diagnostics():
    hardware = create_hardware()

    hardware.gpus[0].vram_used_bytes = (
        10 * 1024**3
    )
    hardware.gpus[0].vram_free_bytes = (
        2 * 1024**3
    )

    model = ModelInfo(
        name="diagnostic-test",
        parameters_b=8.0,
        quantization="Q4_K_M",
        context_length=65536,
    )

    profile = create_recommender().recommend(
        hardware,
        model,
        65536,
    )

    assert profile.fit_status == "not_fit"
    assert profile.execution_strategy == (
        "cpu_offload_or_multi_gpu"
    )
    assert profile.available_vram_bytes == (
        2 * 1024**3
    )
    assert profile.estimated_memory_bytes > 0



def test_num_thread_follows_cpu_logical_cores():
    hardware = create_hardware()
    hardware.cpu.logical_cores = 16

    model = ModelInfo(
        name="thread-test",
        parameters_b=8.0,
        quantization="Q4_K_M",
        context_length=8192,
    )

    profile = create_recommender().recommend(
        hardware,
        model,
        8192,
    )

    assert profile.as_dict()["num_thread"] == 16
