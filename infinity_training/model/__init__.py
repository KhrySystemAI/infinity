from torch import accelerator

device = accelerator.current_accelerator().type if accelerator.is_available() else "cpu" # type: ignore
print(f"Using {device} device")