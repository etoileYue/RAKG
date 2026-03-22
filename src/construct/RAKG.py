import os

LOG_NAME_ENV_KEY = "RAKG_LOGGER_NAME"
DEFAULT_LOGGER_NAME = "AgentLog_merge_test"

# Let the entry point control the shared logger name used by dependent modules.
os.environ.setdefault(LOG_NAME_ENV_KEY, DEFAULT_LOGGER_NAME)

from src.kgAgent import NER_Agent

if __name__ == "__main__":
    json_path = "./data/raw/MINE_merge_test/MINE_test2.json"
    output_dir = "./data/test_merge/processed"
    # skip_ner_list = [1]
    # skip_sim_list = [1]
    ner_agent = NER_Agent()
    ner_agent.process_all_topics(
        json_path=json_path,
        output_dir=output_dir,
        # skip_ner_list=skip_ner_list,
        # skip_sim_list=skip_ner_list,
        existing_kg="data/test_merge/processed/RAKG_graph_re/1.json"
    )