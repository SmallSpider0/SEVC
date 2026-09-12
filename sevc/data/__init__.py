"""Selected public interfaces, loaded without unrelated dependencies."""
from importlib import import_module
_EXPORTS = {'DatasetBundle': ('sevc.data.datasets', 'DatasetBundle'), 'IndexedSubset': ('sevc.data.comparative_partitions', 'IndexedSubset'), 'balanced_subset_indices': ('sevc.data.partition', 'balanced_subset_indices'), 'dataset_targets': ('sevc.data.partition', 'dataset_targets'), 'exact_processed_event_indices': ('sevc.data.processed_events', 'exact_processed_event_indices'), 'indices_sha256': ('sevc.data.partition', 'indices_sha256'), 'load_dataset_bundle': ('sevc.data.datasets', 'load_dataset_bundle'), 'loaders_from_partitions': ('sevc.data.partition', 'loaders_from_partitions'), 'make_synthetic_bundle': ('sevc.data.partition', 'make_synthetic_bundle'), 'materialize_locked_e2_partitions': ('sevc.data.comparative_partitions', 'materialize_locked_e2_partitions'), 'materialize_locked_e2_partitions_for_dataset': ('sevc.data.comparative_partitions', 'materialize_locked_e2_partitions_for_dataset'), 'materialize_prospective_e2_partition': ('sevc.data.comparative_partitions', 'materialize_prospective_e2_partition'), 'partition_dataset': ('sevc.data.partition', 'partition_dataset'), 'partition_index_pool': ('sevc.data.partition', 'partition_index_pool'), 'prospective_target_capacities': ('sevc.data.comparative_partitions', 'prospective_target_capacities'), 'stratified_subset_indices': ('sevc.data.partition', 'stratified_subset_indices'), 'subset_loader': ('sevc.data.partition', 'subset_loader')}
__all__ = list(_EXPORTS)
def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value
def __dir__():
    return sorted(set(globals()) | set(__all__))
