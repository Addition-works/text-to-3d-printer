@echo off
REM create-gpu-vm.bat - Create a GPU VM on Google Compute Engine for Trellis 2
REM
REM Trellis 2 requires a GPU with at least 24GB VRAM.
REM Recommended: NVIDIA L4 (24GB) or A100 (40/80GB)
REM
REM This script tries multiple zones to find available GPU capacity.
REM
setlocal

echo ==========================================
echo  Creating GPU VM for text-to-3d-printer
echo ==========================================
echo.
echo Requirements:
echo   - GPU with 24GB+ VRAM (L4, A100, or similar)
echo   - 200GB boot disk for Docker image
echo   - Ubuntu 22.04 LTS
echo.

REM Try L4 GPUs first (24GB VRAM, good price/performance)
echo Trying NVIDIA L4 GPU (24GB VRAM)...
for %%z in (us-west1-a us-west1-b us-west1-c us-west4-a us-west4-c us-east1-b us-east1-c us-east1-d us-east4-a us-east4-c us-central1-c europe-west1-b europe-west1-c europe-west4-a europe-west4-b europe-west4-c asia-east1-a asia-east1-c asia-northeast1-a asia-northeast1-c) do (
    echo Trying zone: %%z
    gcloud compute instances create text-to-3d-demo ^
        --project=text-to-3d-printer ^
        --zone=%%z ^
        --machine-type=g2-standard-8 ^
        --accelerator=type=nvidia-l4,count=1 ^
        --boot-disk-size=200GB ^
        --image-family=ubuntu-2204-lts ^
        --image-project=ubuntu-os-cloud ^
        --maintenance-policy=TERMINATE
    if not errorlevel 1 (
        echo.
        echo SUCCESS: Created VM in %%z with NVIDIA L4
        goto :done
    )
)

REM Fallback: Try A100 GPUs (40GB VRAM, more expensive but more available)
echo.
echo L4 not available. Trying NVIDIA A100 (40GB VRAM)...
for %%z in (us-central1-a us-central1-b us-central1-c us-central1-f us-east1-b us-east1-c us-west1-b us-west4-b europe-west4-a europe-west4-b asia-east1-c) do (
    echo Trying zone: %%z
    gcloud compute instances create text-to-3d-demo ^
        --project=text-to-3d-printer ^
        --zone=%%z ^
        --machine-type=a2-highgpu-1g ^
        --accelerator=type=nvidia-tesla-a100,count=1 ^
        --boot-disk-size=200GB ^
        --image-family=ubuntu-2204-lts ^
        --image-project=ubuntu-os-cloud ^
        --maintenance-policy=TERMINATE
    if not errorlevel 1 (
        echo.
        echo SUCCESS: Created VM in %%z with NVIDIA A100
        goto :done
    )
)

echo.
echo FAILED: No zones had available GPU capacity
echo.
echo Suggestions:
echo   1. Try again later (capacity fluctuates)
echo   2. Request quota increase for your project
echo   3. Try a different region
echo.
goto :end

:done
echo.
echo ==========================================
echo  VM Created Successfully!
echo ==========================================
echo.
echo Next steps:
echo   1. Open firewall:
echo      gcloud compute firewall-rules create allow-8080 --allow=tcp:8080 --target-tags=http-server
echo      gcloud compute instances add-tags text-to-3d-demo --tags=http-server --zone=ZONE
echo.
echo   2. SSH into VM:
echo      gcloud compute ssh text-to-3d-demo --zone=ZONE
echo.
echo   3. Install Docker and NVIDIA runtime (see README.md)
echo.
echo   4. Build and run the container
echo.

:end
endlocal
