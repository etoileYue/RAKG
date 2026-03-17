import os

LOG_NAME_ENV_KEY = "RAKG_LOGGER_NAME"
DEFAULT_LOGGER_NAME = "AgentLog_test"

# Let the entry point control the shared logger name used by dependent modules.
os.environ.setdefault(LOG_NAME_ENV_KEY, DEFAULT_LOGGER_NAME)

from src.kgAgent import NER_Agent

if __name__ == "__main__":
    json_path = "./data/raw/MINE_test.json"
    output_dir = "./data/test/processed/RAKG_graph_re"
    skip_ner_list = [2]
    ner_agent = NER_Agent()
    ner_agent.process_all_topics(
        json_path=json_path,
        output_dir=output_dir,
        done_offset=2,
        skip_ner_list=skip_ner_list,
        ner_output_dir="./data/test/processed/llmasjudge/ner_data",
        rel_output_dir="./data/test/processed/llmasjudge/rel_data",
    )