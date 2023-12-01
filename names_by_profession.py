import sqlite3
import pandas as pd
import numpy as np

## name, count, grouping

class NamesByProfession:
    def __init__(self, db_path='names.db'):
        try:
            self.con = sqlite3.connect(db_path)
            self.cursor = self.con.cursor()

            # Fetch data from tables
            self.occupations = np.array(self.cursor.execute('SELECT * FROM occupations').fetchall())
            self.by_state = np.array(self.cursor.execute('SELECT * FROM by_state').fetchall())

        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def get_l2_categories(self, gender):
        try:
            # Execute the SQL statement with parameter binding for gender
            sql_statement = 'SELECT level2_main_occ, COUNT(DISTINCT name) FROM occupations WHERE gender = ? GROUP BY level2_main_occ ORDER BY 2 DESC'
            result = pd.read_sql_query(sql_statement, self.con, params=[gender])
            return result
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def get_l3_categories(self, profession, gender):
        try:
            # Execute the SQL statement with parameter binding for both profession and gender
            sql_statement = 'SELECT level3_main_occ, COUNT(DISTINCT name) FROM occupations WHERE level2_main_occ = ? AND gender = ? GROUP BY level3_main_occ ORDER BY 2 DESC'
            result = pd.read_sql_query(sql_statement, self.con, params=[profession, gender])
            return result
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def get_filtered_names(self, gender, profession, state):
        try:
            # Execute SQL statement to get distinct names for the given gender and profession
            sql_statement = 'SELECT name, COUNT(*) as name_count FROM occupations WHERE level3_main_occ = ? AND gender = ? GROUP BY name ORDER BY name_count DESC LIMIT 50'
            result_names = pd.read_sql_query(sql_statement, self.con, params=(profession, gender))

            # Create a list to store success metrics
            success_metrics = []

            # Retrieve success_metric for each name and state
            for name in result_names['name']:
                sql_statement_state = 'SELECT success_metric FROM by_state WHERE name = ? AND state = ?'
                success_metric = pd.read_sql_query(sql_statement_state, self.con, params=(name, state))

                # If success_metric is found, append it to the list, otherwise append None
                if not success_metric.empty:
                    success_metrics.append(success_metric['success_metric'].iloc[0])
                else:
                    success_metrics.append(None)

            # Add success_metrics column to the result_names DataFrame
            result_names['success_metric'] = success_metrics

            return result_names

        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
            return None

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def close_connection(self):
        if self.con:
            self.con.close()
# Example usage
names_profession = NamesByProfession()

# Get distinct names by level2_main_occ
distinct_l2_categories = names_profession.get_l2_categories('Male')
# Print the result
print(distinct_l2_categories)
# Get distinct names by level3_main_occ
result_academia = names_profession.get_l3_categories('Academia', 'Male')
# Print the result
print(result_academia)
# Get physician names cross-referenced from closestM
result_names = names_profession.get_filtered_names(gender='Male', profession='physician', state = 'CA')
# Print the result
print(result_names)

