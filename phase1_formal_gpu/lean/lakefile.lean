import Lake
open Lake DSL

package «fsot_gpu» where
  -- leanOptions := #[]

@[default_target]
lean_lib «FSOTGPU» where
  roots := #[`Trinary, `GpuMemory]
