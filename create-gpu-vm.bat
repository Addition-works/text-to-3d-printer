@echo off
setlocal

for %%z in (us-west1-a us-west1-b us-west1-c us-west4-a us-west4-c us-east1-b us-east1-c us-east1-d us-east4-a us-east4-c us-central1-c europe-west1-b europe-west1-c europe-west4-a europe-west4-b europe-west4-c) do (
    echo Trying zone: %%z
    gcloud compute instances create text-to-3d-demo --project=text-to-3d-printer --zone=%%z --machine-type=g2-standard-8 --accelerator=type=nvidia-l4,count=1 --boot-disk-size=200GB --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud --maintenance-policy=TERMINATE
    if not errorlevel 1 (
        echo SUCCESS in %%z
        goto :done
    )
)

echo FAILED: No zones had available capacity
goto :end

:done
echo Instance created successfully!

:end
endlocal