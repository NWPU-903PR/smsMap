#pragma once
#include<iostream>
#include<fstream>
#include<iomanip>
#include<cstdlib>
#include<stdio.h>
#include<string.h>
#include<ctype.h>
#include<stdint.h>
#include<time.h>
#include<valarray>
#include<vector>
#include<map>
#include "edlib.h"
#define VERSION  "1.0"
#define MAX_LINE_SIZE 300000
#define MAX_BIN_SWAP 2E9

using namespace std;

template<class TYPE>
class Vector : public vector<TYPE>
{
public:
	Vector() : vector<TYPE>() {}
	Vector(size_t size) : vector<TYPE>(size) {}
	Vector(size_t size, const TYPE & deft) : vector<TYPE>(size, deft) {}

	void Append(const TYPE & item) {
		size_t n = this->size();
		if (n + 1 >= this->capacity()) this->reserve(n + n / 5 + 1);
		this->push_back(item);
	}
	int size()const { return (int)vector<TYPE>::size(); }
};

template<class TYPE>
class NVector
{
public:
	TYPE * items;
	int     size;
	int     capacity;

	NVector() { size = capacity = 0; items = NULL; }
	NVector(int n, const TYPE & v = TYPE()) {
		size = capacity = 0; items = NULL;
		Resize(n, v);
	}
	NVector(const NVector & other) {
		size = capacity = 0; items = NULL;
		if (other.items) {
			Resize(other.size);
			memcpy(items, other.items, other.size * sizeof(TYPE));
		}
	}

	~NVector() { if (items) free(items); }

	int  Size()const { return size; }
	void Clear() {
		if (items)
			free(items);
		size = capacity = 0; items = NULL;
	}

	void Resize(int n, const TYPE & value = TYPE()) {
		if (n == size && capacity > 0) return;
		int i;
		// When resize() is called, probably this is the intended size,
		// and will not be changed frequently.
		if (n != capacity) {
			capacity = n;
			items = (TYPE*)realloc(items, capacity * sizeof(TYPE));
		}
		for (i = size; i<n; i++)
			items[i] = value;
		size = n;
	}
	void Append(const TYPE & item) {
		if (size + 1 >= capacity) {
			capacity = size + size / 5 + 1;
			items = (TYPE*)realloc(items, capacity * sizeof(TYPE));
		}
		items[size] = item;
		size++;
	}

	TYPE& operator[](const int i) {
		//if( i <0 or i >= size ) printf( "out of range\n" );
		return items[i];
	}
	TYPE& operator[](const int i)const {
		//if( i <0 or i >= size ) printf( "out of range\n" );
		return items[i];
	}
};


typedef NVector<int>      VectorInt;
typedef Vector<VectorInt> MatrixInt;

void bomb_error(const char *message);
int print_usage();
int print_usage_genome_library_build();
int print_usage_sequence_position_locate();
int print_usage_sequence_banded_alignment();
string getTime();
struct hash_and_index {
	unsigned long long int hash_value;
	unsigned long long int index_in_genome;
};

struct by_hash_value {
	bool operator()(hash_and_index const &left, hash_and_index const &right) {
		return left.hash_value < right.hash_value;
	}
};


struct Options
{
	size_t  max_memory; // -M: 400,000,000 in bytes
	int     min_length; // -l: 10 bases // minimum length to align
	int     banded_width; // -b: 20 // band width in alignment procedure
	int	max_banded_width;
	int     print;
	int     threads;
	int		match_score;
	int		mismatch_score;//substitution
	int		gap_score;
	int		kmer_length; // the k-mer length for locate
	int		local_strict; // strict local alignment, maybe the aligned sequence length < seq->size
	size_t  max_entries;
	size_t  max_sequences;
	size_t  mem_limit;

	string  input;
	string  genome;
	string  output;
	string  genomeLibFile;
	string  sequencePosFile;
	string  model;  // -m banded: my alignment model, -m edlib :edlib alignment model
					//string  edlib; // edlib alignment model

	Options() {
		max_memory = 800000000;
		min_length = 10;
		kmer_length = 15;
		print = 0;
		threads = 1;
		max_entries = 0;
		max_sequences = 1 << 20;
		mem_limit = 100000000;
		match_score = 2;
		mismatch_score = -2;
		gap_score = -2;
		banded_width = 900;
		model = "edlib";
	};

	bool SetOptionCommon(const char *flag, const char *value);
	bool SetOption(const char *flag, const char *value);
	bool SetOptions(int argc, const char *argv[]);
	void SetKmerlength(int aa);
	void SetMaxBandedWidth(int ww);
	void Validate();
	void Print();
};

struct Sequence {
	// real sequence, if it is not stored swap file:
	string header;
	char *data;
	unsigned int tar_id; // matched reference index 
	unsigned int   size; // length of the sequence
	unsigned int position_in_sequence;
	unsigned int position_in_genome;
	unsigned int position_in_genome_minus;
	short plus; // 1: plus direction, 0: minus direction alignment
	int   bufsize;
	double  identify; // the identify with genome
	FILE *swap;
	// stream offset of the sequence:
	int   offset;
	// stream offset of the description string in the database:
	size_t   des_begin;
	// length of the description:
	int   des_length;
	// length of the description in quality score part:
	int   des_length2;
	// length of data in fasta file, including line wrapping:
	int   dat_length;
	char *identifier; // sequence header
	int id_sorted;// sorted index of the sequence
	int   index;  // index of the sequence in the original database:
	short state;
	int   coverage[4];
	int nMatch;
	int nSubsitute;
	int nDelete;
	int nInsert;
	int nAlignedBase;
	char * aligned1; // self 
	char * middle;   // blasted liked output
	char * aligned2; // genome aligned

	Sequence();
	Sequence(const Sequence & other);
	~Sequence();

	void Clear();

	void operator=(const char *s);
	void operator+=(const char *s);

	void Resize(int n);
	void Reserve(int n);

	void Swap(Sequence & other);
	int Format();
	void Set_aligned_information(string a1, string a2, string midd, int matchn, int subsituten, int insertn, int deleten, int aligned_base_n, double sim);
	void Set_sequence_locate(short direction, unsigned int pos_in_seq, unsigned int pos_in_gen, unsigned int pos_in_gen_minus);
	//void ConvertBases();

	void SwapIn();
	void SwapOut();
	//void PrintInfo(int id, FILE *fin, FILE *fout, const Options & options, char *buf);
	//void PrintInfo(int id, FILE *fin, FILE *fout, FILE *fout2, const Options & options, char *buf);
};

class SequenceDB
{
public:

	Vector<Sequence*>  sequences;
	//Vector<int>        rep_seqs;
	//Vector<Vector<int> > clustersIndex;
	//Vector<Vector<int> > clusters_seeds;
	//Vector<Vector<float> > clusters_similarity;
	//Vector<int> seeds_calculated;
	//Vector<float> cluster_mean;
	//Vector<float> cluster_std;
	long long total_letter;
	long long total_desc;
	size_t max_len;
	float mean_len;
	size_t min_len;
	//size_t len_n50;

	void Clear() {
		for (int i = 0; i < sequences.size(); i++) delete sequences[i];
		sequences.clear(); //rep_seqs.clear();
	}

	SequenceDB() {
		total_letter = 0;
		total_desc = 0;
		min_len = 0;
		max_len = 0;
		//len_n50 = 0;
		mean_len = 0.0;
	}
	~SequenceDB() { Clear(); }

	void Read(const char *file, const Options & options);
	void SequenceStatistic(Options & options);
	void SequencePositionWriteToFile(Options & options);
	void ReadSequencePositionFile(Options & options);
	void WriteAlignedToFile(Options & options);
};

class GenomeDB
{
public:

	Sequence  genome[100];
	Sequence genome_minus_sets[100];
	Sequence *genome_minus;
	int genome_num;
	char * genomee;
	char * genome_minuss;
	Vector<NVector<unsigned int> > genomeIndexLocatePlus; // genome k-mer locations (plus direction)
	Vector<NVector<unsigned int> > genomeIndexLocateMinus; // genome k-mer locations (Minus direction)
	unsigned int genome_length;
	unsigned int * hash_value_each_position_point; // the hash value of each k-mer from 0 to end in the genome
	unsigned int * hash_value_index; // hash value position in the genome

									 // store the hash value index in the sorted hash_value_each_position_point
									 // and not exist: -1
	long long int * kmer_exist_and_index; // 
										  // minus derection
	unsigned int * hash_value_each_position_point_minus;
	unsigned int * hash_value_index_minus;
	long long int * kmer_exist_and_index_minus;
	//unsigned long long *hash_value_each_position; // store the hash value in each position
	//unsigned long long *index_each_hash; // store the corresponding index in the genome
	//vector<unsigned int> hash_value_each_position; // store the hash value and index in the genome
		GenomeDB() {
		genome_length = 0;
	}
	void Read_my(const char *file, const Options & options);
	void GenomeMinus();
	void GenomeIndexBuildWriteToFile(const Options & options);
	void GenomeIndexBuildAndLocate(SequenceDB & seqsDB, const Options & options);
	void GenomeIndexBuildForMulti_threads(unsigned int i, const Options & options);
	int GenomeLibraryRead(const Options & options);
	void SequenceLocateInGenome(Sequence* seq, const Options & options);
	//void SequenceLocateInGenome_merge_sort(Sequence* seq, const Options & options, unsigned int * hash_value_each_position_point, unsigned int * hash_value_index, long long int * kmer_exist_and_index, unsigned int * hash_value_each_position_point_minus, unsigned int * hash_value_index_minus, long long int * kmer_exist_and_index_minus);
	void SequenceLocateInGenome_merge_sort(Sequence* seq, const Options & options);
	void SequenceLocateInGenome_multi_threads(SequenceDB & seq_db, const Options & options);
	void SequenceLocateInGenome_vector(Sequence* seq, const Options & options);
	float seq_align_global_my_no_banded(Sequence *seq, const Options & options);
	float seq_align_local_my_no_banded(Sequence *seq, const Options & options);
	float seq_align_local_my_with_banded(Sequence *seq, const Options & options);
	float seq_align_local_diagonal_banded(Sequence *seq, const Options & options);
	float one_seq_align_local_diagonal_banded_for_multi_threads(Sequence *seq, const Options & options);
	float one_seq_align_edlib_for_multi_threads(Sequence *seq, const Options & options);
	float seq_align_local_diagonal_banded_one_matrix(Sequence *seq, const Options & options, int **matrix, int **track);
	int seq_align_multi_threads(SequenceDB & seqsDB, const Options & options);

};

void printAlignment(const char* query, const char* target,
	const unsigned char* alignment, const int alignmentLength,
	const int position, const EdlibAlignMode modeCode);
