#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <boost/json.hpp>

using namespace std;

/* DISCLAIMER
	No one has copyrighted this software
*/
/* RULES
	WE LOVE BOOST LIBS. ANY Boost hate is disliked.
	If you can remove iteration, you get a "raise."
	The lifeblood of this program is std::unordered_map.
*/
/* WARNING
	For some ridiculous reason, rdbuf() DESTROYS buffer. DO NOT TRY TO ACCESS, IT LEADS TO ABORT()! I hate the STL :(
*/

/* NOTICE 
	GUYS THIS STRUCT WAS MADE BY A CLANKER
*/
struct pair_hash {
	template <class T1, class T2>
	size_t operator()(const pair<T1, T2>& p) const {
		auto h1{ hash<T1> {} (p.first) };
		auto h2{ hash<T2> {} (p.second) };
		return h1 ^ (h2 << 1);
	}
};
/* NOTICE
	No clanker code beyond this point
*/

struct training_dat {
	vector<unordered_map<string, double>> data_set;
	vector<unordered_map<string, bool>> correctness_set;
};

class ai_model {
public:
	ifstream model_json;
	boost::json::object main_obj;

	boost::system::error_code open_model(string path = { "./" }, string name = { "template.json" }) {
		boost::system::error_code ec;
		string formatted_path{ path + name };

		model_json.open(formatted_path, ios::in);
		stringstream buf_access;
		buf_access << model_json.rdbuf();

		boost::json::value model_val{ boost::json::parse(buf_access.str(), ec) };
		main_obj = model_val.as_object();

		return ec;
	}

	static void train(vector<pair<string, string>>& connectors, unordered_map<pair<string, string>, double, pair_hash>& weighters, vector<unordered_map<string, double>> sets_input_activations_func_in, unordered_map<string, float> biasers, vector<unordered_map<string, bool>> correct_vals, long long correctness_threshold) {
		bool good{ false };
		unordered_map<pair<string, string>, double, pair_hash> weights{weighters};
		unordered_map<string, float> biases{ biasers };
		vector<unordered_map<string, double>> sets_input_activations_vec{ sets_input_activations_func_in };
		auto activations_vec_begin{ sets_input_activations_vec.begin() };
		auto activations_vec_end{ sets_input_activations_vec.end() };

		while (!good) {
			unordered_map<string, long long> truth_counters;
			for (int index_counter{ 0 }; index_counter < sets_input_activations_vec.size(); ++index_counter) {
				auto sets_input_activations{ sets_input_activations_vec.at(index_counter) };
				unordered_map<string, double> bounties;
				good = true;

				// Size the counter record
				for (auto& conn_pair : connectors) {
					if (truth_counters.count(conn_pair.first) == 0) {
						truth_counters.emplace(conn_pair.first, 0);
					}
					else {
						ignore;
					}
				}
				// Size the bounty (logits) record
				for (auto& conn_pair : connectors) {
					if (bounties.count(conn_pair.first) == 0) {
						bounties.emplace(conn_pair.first, 0);
					}
					else {
						ignore;
					}
				}
				// Scale the dataset for compatibility
				for (auto& amounter : connectors) {
					if (sets_input_activations.count(amounter.first) < 1) {
						sets_input_activations.emplace(amounter.first, 0);
					}
					else if (sets_input_activations.count(amounter.second) < 1) {
						sets_input_activations.emplace(amounter.second, 0);
					}
					else {
						ignore;
					}
				}
				// Find the bounties
				for (auto& finder : connectors) {
					bounties.at(finder.first) += weights.at(pair<string, string> { finder.first, finder.second })* sets_input_activations.at(finder.second);
					cout << "Bounty at " << bounties.at(finder.first) << "\n";
				}
				// Find the output activations
				unordered_map<string, bool> activations;
				for (auto& value : bounties) {
					cout << "Current bias is " << biases.at(value.first) << "\n";
					if (value.second > biases.at(value.first)) {
						activations.emplace(value.first, true);
					}
					else {
						activations.emplace(value.first, false);
					}
					cout << "Value @" << value.first << " is a binary of " << (--activations.end())->second << "\n";
				}

				// Determine next steps
				for (auto& node_truth_pair : correct_vals.at(index_counter)) {
					if (activations.at(node_truth_pair.first) == node_truth_pair.second && node_truth_pair.second == true) {
						++truth_counters.at(node_truth_pair.first);
						cout << node_truth_pair.first << " Made it's iteration " << truth_counters.at(node_truth_pair.first) << ".\n";
					}

					else if (activations.at(node_truth_pair.first) == node_truth_pair.second && node_truth_pair.second == false) {
						++truth_counters.at(node_truth_pair.first);
						cout << node_truth_pair.first << " Made it's iteration " << truth_counters.at(node_truth_pair.first) << ".\n";
					}

					else if (activations.at(node_truth_pair.first) != node_truth_pair.second && node_truth_pair.second == true) {
						// Add activations to weights
						cout << "Because of " << node_truth_pair.first << " we are adjusting weights +\n";
						for (auto& weigh : weights) {
							if (weigh.first.first == node_truth_pair.first) {
								weights.at(weigh.first) += sets_input_activations.at(weigh.first.second);
							}
							else {
								ignore;
							}
						}
						/* OLD SCRIPT (FOR REFERENCE)
						for (auto weigh : weights) {
							weights.at(weigh.first) += sets_input_activations.at(weigh.first);
						} */
						
						// Down the biases a bit
						cout << "Adjusting biases...\n";
						for (auto& bias : biases) {
							float* current_bias{ &biases.at(bias.first) };
							*current_bias -= 0.015;
							*current_bias = round(*current_bias * 10000) / 10000;
						}
					}

					else if (activations.at(node_truth_pair.first) != node_truth_pair.second && node_truth_pair.second == false) {
						// Subtract activations from weights
						cout << "Because of " << node_truth_pair.first << " we are adjusting weights -\n";
						for (auto& weigh : weights) {
							if (weigh.first.first == node_truth_pair.first) {
								weights.at(weigh.first) -= sets_input_activations.at(weigh.first.second);
							}
							else {
								ignore;
							}
						}
						/* OLD SCRIPT (FOR REFERENCE)
						for (auto weigh : weights) {
							weights.at(weigh.first) -= sets_input_activations.at(weigh.first);
						} */
						
						// Up the biases a bit
						cout << "Adjusting biases...\n";
						for (auto& bias : biases) {
							float* current_bias{ &biases.at(bias.first) };
							*current_bias += 0.015;
							*current_bias = round(*current_bias * 10000) / 10000;
						}
					}
				}

				// PLEASE keep in mind the amount of training data vs. the truth count
				for (auto& entry : truth_counters) {
					if (entry.second < correctness_threshold) {
						good = good && false;
					}
					else {
						ignore;
					}
				}
			}
		}

		cout << "\n~>Finished. Weights are: ";
		for (auto& weight : weights) {
			cout << "\n" << weight.first.second << ": " << weight.second << ";";
		}
		cout << "\n~>Biases are ";
		for (auto& bias : biases) {
			cout << "\n" << bias.first << ": " << bias.second << ";";
		}

		return;
	}
};

pair<unordered_map<string, boost::json::object>, unordered_map<string, boost::json::object>> parse_nodes(boost::json::object input_object) {
	/* NOTE
		This function exists to take the model, separate its nodes, and store the values of each into a pair of maps. FIRST VAL IN PAIR IS INPUT MAP, SECOND IS OUTPUT MAP!
	*/
	boost::json::object parse_object{input_object};
	unordered_map<string, boost::json::object> node_mapping;

	for (boost::json::key_value_pair node : parse_object) {
		auto node_key{ node.key() };
		if (node_key == "train_dat" || node_key == "correct_vals" || node_key == "amount_correct_needed") {
			ignore;
		}
		else {
			node_mapping.emplace(node_key, node.value().as_object());
		}
	}

	unordered_map<string, boost::json::object> input_node_mapping;
	unordered_map<string, boost::json::object> output_node_mapping;
	/*unordered_map<string, boost::json::object> hidden_node_mapping;*/

	for (auto& elem : node_mapping) {
		if (elem.second.at("flag") == "input") {
			input_node_mapping.emplace(elem.first, elem.second);
		}
		else if (elem.second.at("flag") == "output") {
			output_node_mapping.emplace(elem.first, elem.second);
		}
		//else if (elem.second.at("flag") == "hidden") {
		//	hidden_node_mapping.emplace(elem.first, elem.second);
		//}
		else {
			abort();
		}
	}

	pair<unordered_map<string, boost::json::object>, unordered_map<string, boost::json::object>> return_val_node_mapping{ input_node_mapping, output_node_mapping };
	return return_val_node_mapping;
}

vector<pair<string, string>> parse_connections(unordered_map<string, boost::json::object> input_output_mappings) {
	/* NOTE
		This function takes the node mapping and converts it into a mapping of node connections
	*/
	unordered_map<string, boost::json::object> parse_object{ input_output_mappings };
	vector<pair<string, string>> connections_mapping;

	for (auto& out_node : parse_object) {
		boost::json::array current_connection_array{ out_node.second["connections"].as_array() };
		for (auto& current_connection : current_connection_array) {
			connections_mapping.push_back(pair<string, string> {out_node.first, current_connection.as_string()});
		}
	}

	return connections_mapping;
}

unordered_map<pair<string, string>, double, pair_hash> parse_weights(unordered_map<string, boost::json::object> input_val_map, vector<pair<string, string>> connection_mapping) {
	/* NOTE
		This function blah blah blah... Just kidding. This function take the nodes and maps weights to them w/ everyone's favorite data structure: the unordered map!!
	*/
	unordered_map<string, boost::json::object> parse_weight_object{ input_val_map };
	unordered_map<pair<string, string>, double, pair_hash> weights_mapping;

	for (auto& node_dat_pair : parse_weight_object) {
		boost::json::object weights_as_objects{ node_dat_pair.second.at("weights").as_object() };
		for (auto& node_weight_pair : weights_as_objects) {
			weights_mapping.emplace(pair<string, string> {node_dat_pair.first, node_weight_pair.key_c_str()}, node_weight_pair.value().as_double());
		}
	}

	return weights_mapping;
}

unordered_map<string, float> parse_bias(unordered_map<string, boost::json::object> out_map) {
	unordered_map<string, float> bias_list;

	for (auto& item : out_map) {
		bias_list.emplace(item.first, item.second.at("bias").as_double());
	}

	return bias_list;
}

training_dat datanizer(boost::json::object root) {
	// Grab the training data
	training_dat dat_set;
	boost::json::object training_data_extract{ root.at("train_dat").as_object() };
	cout << "\n   |  Targeted training data";
	for (auto& pair : training_data_extract) {
		unordered_map<string, double> tmp_node_set;
		auto current_set{ pair.value().as_object() };
		for (auto& current_node_dat : current_set) {
			tmp_node_set.emplace(current_node_dat.key_c_str(), current_node_dat.value().to_number<double>());
		}
		dat_set.data_set.push_back(tmp_node_set);
	}

	// Grab the correct outputs
	boost::json::object correctness_data_extract{ root.at("correct_vals").as_object() };
	cout << "\n   |  Targeted additional data";
	for (auto& set : correctness_data_extract) {
		unordered_map<string, bool> tmp_correct_set;
		for (auto& node : set.value().as_object()) {
			tmp_correct_set.emplace(node.key_c_str(), node.value().as_bool());
		}
		dat_set.correctness_set.push_back(tmp_correct_set);
		if (dat_set.correctness_set.at(stoi(set.key_c_str())) == tmp_correct_set) {
			ignore;
		}
		else {
			cout << "\nERROR HALLLPPPPP";
		}
	}

	return dat_set;
}

int main() {
	string mode_choice{ "\0" };

	cout << "~> Initializing model...";
	ai_model* runtime_model{ new ai_model};
	runtime_model->open_model();
 	
	cout << "\n~> Generating mappings...";
	auto mappings{ parse_nodes(runtime_model->main_obj) };
	unordered_map<string, boost::json::object> input_map{ mappings.first };
	unordered_map<string, boost::json::object> output_map{ mappings.second };
	cout << "\n   |\n  Nodes mapped";

	vector<pair<string, string>> connection_map{ parse_connections(output_map) };
	cout << "\n  Connections mapped";
	unordered_map<pair<string, string>, double, pair_hash> weights_map{ parse_weights(output_map, connection_map) };
	cout << "\n  Weights mapped";
	unordered_map<string, float> bias_mapping{ parse_bias(output_map) };
	cout << "\n  Biases mapped";
	cout << "\n  Crunching training data";
	training_dat input_training_data{ datanizer(runtime_model->main_obj) };
	cout << "\n  Training data crunched" << "\n   |";

	cout << "\n~> Finished setup. Please type your mode (\"Training\" or \"Running\")> ";
	cin >> mode_choice;

	if (mode_choice == "Training") {
		ai_model::train(connection_map, weights_map, input_training_data.data_set, bias_mapping, input_training_data.correctness_set, runtime_model->main_obj.at("amount_correct_needed").as_int64());
	}

	return 0;
}