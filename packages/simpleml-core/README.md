# SimpleML-Core

SimpleML-Core represents the core of SimpleML. It intentionally maintains minimal dependencies and implements the abstract structure of the SimpleML ecosystem. Core is not meant to be used on its own. It should be bundled ala carte with the specific namespace packages to define a full deployment.

# Abstract Core

- Persistables: Datasets, Pipelines, Models, Metrics
- Transformer Base
- ORM Bindings (not DB centric)
- Persistence Base (Pickle/Cloudpickle)
- Executors: Native python (main process, threading, multiprocessing)
- Utilities: Hashing, Configuration
- Registries