import os

LOG_NAME_ENV_KEY = "RAKG_LOGGER_NAME"
DEFAULT_LOGGER_NAME = "AgentLog_merge_test"

# Let the entry point control the shared logger name used by dependent modules.
os.environ.setdefault(LOG_NAME_ENV_KEY, DEFAULT_LOGGER_NAME)

from src.kgAgent import NER_Agent

if __name__ == "__main__":
    json_path = "./data/raw/MINE_test_line/MINE_test2.json"
    output_dir = "./data/tmp_merged/processed"
    existing_kg = {1: "data/tmp/processed/RAKG_graph_re/1.json"}

    ner_agent = NER_Agent()
    ner_agent.process_all_topics(
        json_path=json_path,
        output_dir=output_dir,
        # skip_cur_ner=True,
        # skip_cur_sim=True,
        existing_kg=existing_kg
    )