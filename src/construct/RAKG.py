import os

LOG_NAME_ENV_KEY = "RAKG_LOGGER_NAME"
DEFAULT_LOGGER_NAME = "AgentLog_short_test"

LOG_FILE_ENV_KEY = "RAKG_LOGGER_FILE"
DEFAULT_LOGGER_FILE = "Agent3.28.log"

DEBUG_FILE_ENV_KEY = "RAKG_DEBUG_FILE"
DEFAULT_DEBUG_FILE = "Debug3.28.log"
# Let the entry point control the shared logger name used by dependent modules.
os.environ.setdefault(LOG_NAME_ENV_KEY, DEFAULT_LOGGER_NAME)
os.environ.setdefault(LOG_FILE_ENV_KEY, DEFAULT_LOGGER_FILE)
os.environ.setdefault(DEBUG_FILE_ENV_KEY, DEFAULT_DEBUG_FILE)

from src.kgAgent import NER_Agent

if __name__ == "__main__":
    json_path = "./data/raw/MINE_short10.json"
    output_dir = "./data/new_test/processed"
    force_rebuild = False

    skip_construct = set([1,2, 5, 6, 7, 8, 10])
    # skip_ner_set = set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # skip_sim_set = set([2, 3, 4, 7, 9])
    # skip_rel_set = set([2, 3, 4, 7, 9])
    # existing_kg = {1: "data/tmp/processed/RAKG_graph_re/1.json"}

    ner_agent = NER_Agent()
    ner_agent.process_all_topics(
        json_path=json_path,
        output_dir=output_dir,
        force_rebuild=force_rebuild,
        skip_construct=skip_construct,
        # skip_ner_set=skip_ner_set,
        # skip_sim_set=skip_sim_set,
        # skip_rel_set=skip_rel_set,        
        # existing_kg=existing_kg
    )
