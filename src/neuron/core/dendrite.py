from typing import Any, Iterable, Optional
import torch


class Dendrite:
    def __init__(self, synapse_indices: Optional[Iterable[Any]] = None, device: str = "cuda"):
        self.device = torch.device(
            device if device == "cpu" or torch.cuda.is_available()
            else "cpu"
        )

        # Khai báo rõ ràng _indices có thể là list hoặc None
        self._indices: Optional[list[int]] = (
            list(map(int, synapse_indices))
            if synapse_indices is not None
            else []
        )

        # Khai báo rõ ràng synapse_indices có thể là Tensor hoặc None
        self.synapse_indices: Optional[torch.Tensor] = None
        self._built = False

    def attach(self, synapse_idx: int):
        if self._built or self._indices is None:
            raise RuntimeError("Cannot attach after build().")

        # Thêm kiểm tra này để Pylance chắc chắn self._indices không phải là None
        self._indices.append(int(synapse_idx))
        return synapse_idx

    def build(self):
        if self._indices is None:
            return self

        self.synapse_indices = torch.tensor(
            self._indices,
            dtype=torch.long,
            device=self.device
        )

        self._indices = None
        self._built = True
        return self

    def current(self, synapse_outputs: torch.Tensor):
        if not self._built or self.synapse_indices is None:
            raise RuntimeError("Call build() before current().")

        # Tạo một biến cục bộ để loại bỏ trạng thái Optional của thuộc tính class
        indices_tensor = self.synapse_indices

        if indices_tensor.numel() == 0:
            return synapse_outputs.new_zeros(())

        indices = indices_tensor.to(synapse_outputs.device)
        return synapse_outputs.index_select(0, indices).sum()

    def incoming_synapses(self) -> torch.Tensor:
        if not self._built:
            # Nếu chưa build, self._indices chắc chắn là một danh sách hợp lệ
            assert self._indices is not None, "Internal error: _indices cannot be None before build."
            return torch.tensor(
                self._indices,
                dtype=torch.long,
                device=self.device
            )

        # Nếu đã build, self.synapse_indices chắc chắn đã là một Tensor
        assert self.synapse_indices is not None, "Internal error: synapse_indices cannot be None after build."
        return self.synapse_indices
