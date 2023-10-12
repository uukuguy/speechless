#!/usr/bin/env pthon
"""### Instructions:
Your task is to convert a question into a SQL query, given a Postgres database schema.
Adhere to these rules:\n- **Deliberately go through the question and database schema word by word** to appropriately answer the question
- **Use Table Aliases** to prevent ambiguity. For example, `SELECT table1.col1, table2.col1 FROM table1 JOIN table2 ON table1.id = table2.id`.
- When creating a ratio, always cast the numerator as float

### Input:
 
Generate a SQL query that answers the question `Which airlines offer flights from Chicago (ORD) to New York (JFK), and how many stops do they have, sorted by number of stops?`.
This query will run on a database whose schema is represented in this string:

```
CREATE TABLE flight (
    stops bigint, --Number of stops during the flight
    connections bigint, --Number of connections during the flight
    airline_code text, --Code assigned to airline
    from_airport text, --The 3-letter airport code for the departure location
    to_airport text, --The 3-letter airport code for the arrival location
    airline_flight text, --Code assigned to the flight
    airline_code text, --Code assigned to the airline
    )

-----------

CREATE TABLE airline (
    note text, --Any additional notes or comments about the airline
    airline_name text, --The name of the airline
    airline_code text, --Code assigned to airline
    airline_name text, --The airline's name
    airline_code text, --The code assigned to the airline
    )

-----------

CREATE TABLE flight_stop (
    departure_flight_number bigint, --Flight number of the departing flight from this stop
    stop_airport text, --The 3-letter airport code of the stop
    departure_airline text, --The airline code of the departing flight
    arrival_flight_number bigint, --Flight number of the arriving flight at this stop
    stop_number bigint, --The order of the stop in the flight
    arrival_airline text, --The airline code of the arriving flight
    stop_days text, --Days of the week on which the flight stops at this airport
    departure_time bigint, --The time at which the flight departs from this stop
    stop_time bigint, --The duration of the stop at this airport
    arrival_time bigint, --The time at which the flight arrives at this stop
    stop_airport text, --The 3-letter airport code for the stop
    )

-----------

CREATE TABLE restriction (\n  stopovers text, --The number of stopovers allowed for the fare.\n)

-----------

CREATE TABLE fare (\n  from_airport text, --The 3-letter airport code for the departure location\n  to_airport text, --The 3-letter airport code for the arrival location\n  fare_airline text, --The airline's name\n)

-----------

CREATE TABLE dual_carrier (\n  high_flight_number bigint, --The highest flight number in the range of flights operated by the dual carrier\n  low_flight_number bigint, --The lowest flight number in the range of flights operated by the dual carrier\n  main_airline text, --The name of the main airline operating the flight\n)

-----------

CREATE TABLE airport_service (\n  miles_distant bigint, --The distance of the airport from the city in miles\n  city_code text, --The city code where airport service is available\n  airport_code text, --The airport code where airport service is available\n  city_code text, --Th
e code of the city where the airport is located\n  airport_code text, --The code of the airport\n)

-----------

CREATE TABLE airport (\n  airport_name text, --The name of the airport\n  airport_code text, --The 3-letter airport code\n  airport_location text, --The airport location\n  country_name text, --The country the airport is located in\n  state_code text, --The state the airport is located in\n  state_code text, --The code assigned to the state where the airport is located.\n  country_name text,
--The name of the country where the airport is located.\n  time_zone_code text, --The code assigned to the time zone where the airport is located.\n  airport_code text, --The code assigned to the airport.\n)

-----------

CREATE TABLE city (\n  city_code text, --The city code\n  city_name text, --The city name\n  state_code text, --The state code\n  country_name text, --The country name\n  state_code text, --The code assigned to the state where the city is located\n)\n-----------\n
CREATE TABLE state (\n state_code text, --The state code\n  state_name text, --The state name\n  country_name text, --The country name\n)

-----------

CREATE TABLE ground_service (\n  city_code text, --The city code where ground service is available\n  airport_code text, --The airport code where ground service is available\n  city_code text, --The code for the city where the ground service is provided\n  airport_code text, --The 3-letter code for the airport where the ground service is provided\n)

-----------
```

Additionally, the following are tables/column pairs that can be joined in this database:

```
airline.airline_code can be joined with flight.airline_code
airline.airline_code can be joined with flight_stop.departure_airline
airline.airline_code can be joined with flight_stop.arrival_airline
airport.state_code can be joined with city.state_code
airport.country_name can be joined with city.state_code
airport.time_zone_code can be joined with city.state_code
airport.airport_code can be joined with fare. from_airport
airport.airport_code can be joined with fare.to_airport
airport.airport_code can be joined with flight_stop.stop_airport
airport_service.city_code can be joined with ground_service.city_code
airport_service.airport_code can be joined with ground_service.airport_code
```


### 
Response:Based on your instructions, here is the SQL query I have generated to answer the question `Which airlines offer flights from Chicago (ORD) to New York (JFK), and how many stops do they have, sorted by number of stops?`:
```sql"""

import json
import pandas as pd
import numpy as np
from tqdm import tqdm


def creating_schema(DATASET_JSON):
    schema_df = pd.read_json(DATASET_JSON)
    # schema_df = schema_df.drop(['column_names','table_names'], axis=1)
    schema = []
    f_keys = []
    p_keys = []
    for index, row in schema_df.iterrows():
        tables = row['table_names_original']
        col_names = row['column_names_original']
        col_descs = row['column_names']
        col_types = row['column_types']
        foreign_keys = row['foreign_keys']
        primary_keys = row['primary_keys']
        for col, col_type, col_desc in zip(col_names, col_types, col_descs):
            index, col_name = col
            if index == -1:
                for table in tables:
                    schema.append([row['db_id'], table, '*', 'text', ''])
            else:
                _, col_desc = col_desc
                schema.append([row['db_id'], tables[index], col_name, col_type, col_desc])
        # print(f"{type(col_names)=}, {col_names=}")
        for primary_key in primary_keys:

            # index, column = col_names[primary_key]
            # p_keys.append([row['db_id'], tables[index], column])
            if isinstance(primary_key, list):
                # 多字段主键（BIRD-SQL）
                p_key = []
                for key in primary_key:
                    index, column = col_names[key]
                    p_key.append(column)
                p_keys.append([row['db_id'], tables[index], p_key])
            else:
                # 只有单一主键字段
                index, column = col_names[primary_key]
                p_keys.append([row['db_id'], tables[index], column])

        for foreign_key in foreign_keys:
            first, second = foreign_key
            first_index, first_column = col_names[first]
            second_index, second_column = col_names[second]
            f_keys.append([row['db_id'], tables[first_index], tables[second_index], first_column, second_column])
    spider_schema = pd.DataFrame(
        schema, columns=['Database name', 'Table Name', 'Field Name', 'Field Type', 'Field Description']
    )
    spider_primary = pd.DataFrame(p_keys, columns=['Database name', 'Table Name', 'Primary Key'])
    spider_foreign = pd.DataFrame(
        f_keys,
        columns=[
            'Database name', 'First Table Name', 'Second Table Name', 'First Table Foreign Key',
            'Second Table Foreign Key'
        ]
    )
    return spider_schema, spider_primary, spider_foreign


def main():
    instruction_template = open("../sql-eval/prompts/nl2sql_instruction.md").read()
    input_template = open("../sql-eval/prompts/nl2sql_input.md").read()
    response_template = open("../sql-eval/prompts/nl2sql_response.md").read()

    spider_schema, spider_primary, spider_foreign = creating_schema('../data/BIRD-SQL/train/train_tables.json')
    train_data_df = pd.read_json('../data/BIRD-SQL/train/train.json')
    # # train_others_df = pd.read_json('../data/BIRD-SQL/dev/dev.json')
    # train_data_df = pd.concat([train_spider_df, train_others_df], ignore_index=True)
    # train_gold_file = "../data/BIRD-SQL/train_gold.sql"

    # spider_train_file = "../data/speechless-spider-8659.jsonl"
    spider_train_file = "../data/speechless-birdsql-9428.jsonl"

    fd_out = open(spider_train_file, 'w')
    for idx, row in tqdm(train_data_df.iterrows()):
        query = row['SQL']
        question = row['question']
        db_id = row['db_id']
        evidence = row['evidence']

        db_tables = spider_schema[spider_schema['Database name'] == db_id]['Table Name'].unique()

        question_tables = []
        # ----- BIRD-SQL -----
        for table_name in db_tables:
            if table_name in query:
                question_tables.append(table_name)  

        # ----- spider -----
        # sql_data = row['sql']
        # table_units = sql_data['from']['table_units']
        # for tu_type, table_idx in table_units:
        #     if tu_type == 'table_unit':
        #         # question_tables.appen(table_idx)
        #         question_tables.append(db_tables[table_idx])

        foreign_schema = ""
        related_tables = []
        for foreign_idx, foreign_row in spider_foreign[spider_foreign['Database name'] == db_id].iterrows():
            if foreign_row['First Table Name'] not in question_tables and foreign_row['Second Table Name'] not in question_tables:
                continue

            related_tables.append(foreign_row['First Table Name'])
            related_tables.append(foreign_row['Second Table Name'])

            first_table_nme = foreign_row['First Table Name']
            first_table_foreign_key = foreign_row['First Table Foreign Key']
            second_table_name = foreign_row['Second Table Name']
            second_table_foreign_key = foreign_row['Second Table Foreign Key']

            #airline.airline_code can be joined with flight.airline_code
            foreign_schema += f"{first_table_nme}.{first_table_foreign_key} can be joined with {second_table_name}.{second_table_foreign_key}\n"

        related_tables = list(set(related_tables))
        question_all_tables = list(set(question_tables + related_tables))

        question_schema = "```\n"
        for table_name in question_all_tables:
            question_create_sql = f"CREATE TABLE {table_name} (\n"
            question_fields = spider_schema[(spider_schema['Database name'] == db_id)
                                            & (spider_schema['Table Name'] == table_name)]
            for _, field in question_fields.iterrows():
                field_name = field['Field Name']
                if field_name == "*":
                    continue
                field_type = field['Field Type']
                field_desc = field['Field Description']
                s = f'  {field_name} {field_type}, --{field_desc}\n'
                question_create_sql += s
            question_create_sql += ')\n'
            question_create_sql += "-----------\n"
            question_schema += question_create_sql
        question_schema += "```\n"

        if len(foreign_schema) > 0:
            question_schema += "\nAdditionally, the following are tables/column pairs that can be joined in this database:\n\n"
            question_schema += "```\n"
            question_schema += foreign_schema
            question_schema += "```\n"

        # prompt_template = open("../sql-eval/prompts/nl2sql_prompt.md").read()
        # prompt = prompt_template.format(user_question=question ,table_metadata_string=question_schema)
        instruction = instruction_template
        input = input_template.format(user_question=question, table_metadata_string=question_schema)
        response = response_template.format(response=query)

        # response = f"\n{query}\n```\n"

        # print(f"{prompt=}")
        # print(prompt)
        # print(f"{response=}")

        question_data = {
            'instruction': instruction,
            'input': input,
            'response': response,
            "category": "birdsql",
            "skip_prompt_formatting": False,
            "system": ""
        }
        fd_out.write(json.dumps(question_data, ensure_ascii=False) + "\n")
    fd_out.close()
    print(f"Saved {len(train_data_df)} questions to {spider_train_file}")


if __name__ == '__main__':
    main()
