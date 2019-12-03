#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <algorithm>
#include <time.h>

//#define NORMALIZE_FEATURES
//#define EVEL_ONE_TO_ONE
#define USE_MIN_DISTANCE

#define MAX_DIST 1e9
#define MAX_EPOCH 3

typedef std::vector<float> FEATURE;
typedef std::vector<FEATURE> PERSON;

PERSON readPersonFromFile(std::string f)
{
	PERSON person;
	std::ifstream infile;
	infile.open(f);
	assert(infile.is_open());
	std::string s;
	while (getline(infile, s))
	{
		FEATURE feat;
		std::istringstream iss(s);
		std::string t;
		while (std::getline(iss, t, ' ')) {
			float val;
			std::istringstream tr(t);
			tr >> val;
			feat.push_back(val);
		}
		person.push_back(feat);
		if (feat.size() != 256)
		{
			std::cout << "Feature size wrong! Current size:" << feat.size() << std::endl;
		}
	}
	infile.close();
	return person;
}

std::vector<size_t> generateRandIds(size_t max_id, size_t num_ids)
{
	assert(max_id >= num_ids);

	std::vector<size_t> id_pool;
	std::vector<size_t> ids;
	for (int i = 0; i < max_id; i++)
	{
		id_pool.push_back(i);
	}
	for (int i = 0; i < num_ids; i++)
	{
		auto it = id_pool.begin() + (rand() % id_pool.size());
		ids.push_back(*it);
		id_pool.erase(it);
	}
	return ids;
}

std::vector<size_t> generateDiffIds(size_t max_id, std::vector<size_t> ids_in, int max_out_length = 1000)
{
	std::vector<size_t> id_pool;
	std::vector<size_t> ids;

	for (int i = 0; i < max_id; i++)
	{
		id_pool.push_back(i);
	}
	std::sort(ids_in.begin(), ids_in.end());
	std::set_difference(id_pool.begin(), id_pool.end(), ids_in.begin(), ids_in.end(), std::inserter(ids, ids.begin()));
	if (ids.size() > max_out_length)
	{
		std::vector<size_t> ids_prune;
		for (int i = 0; i < max_out_length; i++)
		{
			ids_prune.push_back(ids[i]);
		}
		ids = ids_prune;
	}
	return ids;
}

float cosineDist(std::vector<float> vec1, std::vector<float> vec2)
{
	assert(vec1.size() == vec2.size());
	float mod1 = 0, mod2 = 0, product = 0;
	for (int i = 0; i < vec1.size(); i++)
	{
		mod1 += vec1[i] * vec1[i];
		mod2 += vec2[i] * vec2[i];
		product += vec1[i] * vec2[i];
	}
	return (1 - product / sqrt(mod1 * mod2));
}

#ifdef USE_MIN_DISTANCE
float manyToManyDist(std::vector<FEATURE> feats1, std::vector<FEATURE> feats2)
{
	//assert(feats1.size() == feats2.size());
	float min_dist = MAX_DIST;
	for (auto feat1 : feats1)
	{
		for (auto feat2 : feats2)
		{
			float dist = cosineDist(feat1, feat2);
			if (dist < min_dist) min_dist = dist;
		}
	}
	return min_dist;
}
#else
float manyToManyDist(std::vector<FEATURE> feats1, std::vector<FEATURE> feats2)
{
	//assert(feats1.size() == feats2.size());
	float dist_sum = 0;
	int count = 0;
	for (auto feat1 : feats1)
	{
		for (auto feat2 : feats2)
		{
			float dist = cosineDist(feat1, feat2);
			dist_sum += dist;
			count++;
		}
	}
	return dist_sum/count;
}
#endif

void Normalize(FEATURE& feat)
{
	float mod = 0;
	for (int i = 0; i < feat.size(); i++)
	{
		mod += feat[i] * feat[i];
	}
	mod = sqrt(mod);
	for (int i = 0; i < feat.size(); i++)
	{
		feat[i] = feat[i]/mod;
	}
}

int main()
{
	// Read persons feature data from txt files.
	std::cout << "Start globbing... wait..." << std::endl;
	std::vector<PERSON> persons;
	std::vector<cv::String> file_paths;
	cv::glob("../data", file_paths); 
	std::cout << "Start reading person data." << std::endl;
	for (auto f : file_paths)
	{
		PERSON person = readPersonFromFile(f);
		persons.push_back(person);
	}
	std::cout << "Reading finished." << std::endl;

#ifdef NORMALIZE_FEATURES
	std::cout << "Start normalize person." << std::endl;
	std::vector<PERSON> persons_normalized;
	for (auto person : persons)
	{
		PERSON person_norm;
		for (auto feature : person)
		{
			Normalize(feature);
			person_norm.push_back(feature);
		}
		persons_normalized.push_back(person_norm);
	}
	persons = persons_normalized;
	std::cout << "Normalize person finished." << std::endl;
#endif

	int epoch = 0;
	int total_correct_count1 = 0, total_wrong_count1 = 0;
	int total_correct_count2 = 0, total_wrong_count2 = 0;
	srand((unsigned)time(NULL));
	while (epoch < MAX_EPOCH)
	{
#ifdef EVEL_ONE_TO_ONE
		// Part1: one-to-one evaluation.
		// Step1: Split person data to gallery and query.
		std::vector<size_t> gallery_feats_index;
		std::vector<size_t> query_feats_index;
		for (auto person : persons)
		{
			std::vector<size_t> rand_ids = generateRandIds(person.size(), 2);
			gallery_feats_index.push_back(rand_ids[0]);
			query_feats_index.push_back(rand_ids[1]);
		}

		// Step2: Find the most similar person in all persons. For each query feature ids, compare with all gallery ids.
		assert(gallery_feats_index.size() == persons.size());
		assert(query_feats_index.size() == persons.size());
		int correct_count = 0, wrong_count = 0;
		for (int i_person = 0; i_person < persons.size(); i_person++)
		{
			size_t query_feat_index = query_feats_index[i_person];
			FEATURE query_feat = persons[i_person][query_feat_index];
			int matched_id = -1;
			float min_dist = MAX_DIST;
			for (int j_person = 0; j_person < persons.size(); j_person++)
			{
				size_t gallery_feat_index = gallery_feats_index[j_person];
				std::vector<float> gallery_feat = persons[j_person][gallery_feat_index];
				float dist = cosineDist(query_feat, gallery_feat);
				if (dist < min_dist)
				{
					min_dist = dist;
					matched_id = j_person;
				}
			}
			if (matched_id == i_person) { std::cout << "."; correct_count++; }
			else { std::cout << "X"; wrong_count++; }
		}
		std::cout << std::endl;
		std::cout << "Correct count(1 to 1):" << correct_count << std::endl;
		std::cout << "Wrong count(1 to 1):" << wrong_count << std::endl;
		std::cout << std::endl;
#else
		int correct_count = 0, wrong_count = 0;
#endif

		// Part2: many-to-many evaluation.
		// Step1: Split person data to gallery and query.
		std::vector<std::vector<size_t>> query_feats_indexes;
		std::vector<std::vector<size_t>> gallery_feats_indexes;
		for (auto person : persons)
		{
			std::vector<size_t> query_ids = generateRandIds(person.size(), 5);
			std::vector<size_t> gallery_ids = generateDiffIds(person.size(), query_ids);
			query_feats_indexes.push_back(query_ids);
			gallery_feats_indexes.push_back(gallery_ids);
		}

		// Part2: Calculate cluster centers for every person.
		const int K = 1;
		std::vector<PERSON> persons_with_K_feats;

		auto id_iter = gallery_feats_indexes.begin();
		for (auto person : persons)
		{
			int feature_dims = person[0].size();
			cv::Mat gallery_data(id_iter->size(), feature_dims, CV_32FC1);

			size_t i_row{ 0 };
			for (auto id : *id_iter)
			{
				cv::Mat person_data = cv::Mat(person[id]).reshape(0, 1);
				person_data.copyTo(gallery_data.rowRange(i_row, i_row + 1));
				i_row++;
			}
			id_iter++;

			PERSON K_centers;
			int attemps = 5;
			const cv::TermCriteria term_criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 300, 0.01);
			cv::Mat labels_, centers_;
			float value = cv::kmeans(gallery_data, K, labels_, term_criteria, attemps, cv::KMEANS_RANDOM_CENTERS, centers_);
			for (int i = 0; i < K; i++)
			{
				//const float* p = centers_.ptr<float>(i);
				//K_centers.push_back(std::vector<float>(p, p + centers_.cols));
				K_centers.push_back((std::vector<float>)centers_.rowRange(i, i + 1));
				//K_centers.push_back((std::vector<float>)gallery_data.rowRange(i, i + 1));// !!!!!!!!!!!!--test for only 1 picture.
			}
			persons_with_K_feats.push_back(K_centers);
			// std::cout << persons_with_K_feats .size() << " person clustered. K-means-value:" << value <<std::endl;
		}

		// Step2: Find the most similar person in all persons. For each query feature ids, compare with all gallery ids.
		assert(gallery_feats_indexes.size() == persons.size());
		assert(query_feats_indexes.size() == persons.size());
		int correct_count2 = 0, wrong_count2 = 0;
		for (int i_person = 0; i_person < persons.size(); i_person++)
		{
			std::vector<size_t> query_feat_indexes = query_feats_indexes[i_person];
			std::vector<FEATURE> query_feats;
			for (auto index : query_feat_indexes)
			{
				query_feats.push_back(persons[i_person][index]);
			}
			int matched_id = -1;
			float min_dist = MAX_DIST;
			for (int j_person = 0; j_person < persons.size(); j_person++)
			{
				std::vector<FEATURE> gallery_K_feats;
				for (auto feature : persons_with_K_feats[j_person])
				{
					gallery_K_feats.push_back(feature);
				}
				float dist = manyToManyDist(query_feats, gallery_K_feats);
				if (dist < min_dist)
				{
					min_dist = dist;
					matched_id = j_person;
				}
			}
			if (matched_id == i_person) { std::cout << "."; correct_count2++; }
			else { std::cout << "X"; wrong_count2++; }
		}
		std::cout << std::endl;
		std::cout << "Correct count(n to n):" << correct_count2 << std::endl;
		std::cout << "Wrong count(n to n):" << wrong_count2 << std::endl;
		std::cout << std::endl;

		// Summary
		epoch++;
		total_correct_count1+= correct_count;
		total_wrong_count1 += wrong_count;
		total_correct_count2 += correct_count2;
		total_wrong_count2 += wrong_count2;
		std::cout << "=============== epoch " << epoch << " summary ================"<<std::endl;
		std::cout << "Total correct (1 to 1): " << total_correct_count1 << ", total wrong (1 to 1): " << total_wrong_count1
			<< ", precision: " << (float)total_correct_count1 / (float)(total_correct_count1 + total_wrong_count1) << std::endl;
		std::cout << "Total correct (n to n): " << total_correct_count2 << ", total wrong (n to n): " << total_wrong_count2
			<< ", precision: " << (float)total_correct_count2 / (float)(total_correct_count2 + total_wrong_count2) << std::endl;
		std::cout << std::endl << std::endl;
	}

	return 0;
}