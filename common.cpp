#include "common.h"
#include <assert.h>
#include <limits.h>
#include <algorithm>
#include <math.h>
#include <string>
#include <cstdio>
#include <sstream>
#include <numeric>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h> 
#include <unistd.h>
#include <sys/mman.h>
#ifndef NO_OPENMP

#include<omp.h>

#define WITH_OPENMP " (+OpenMP)"

#else

#define WITH_OPENMP ""

#define omp_set_num_threads(T) (T = T)
#define omp_get_thread_num() 0

#endif
int min(int x, int y) {
	return (x>y ? y : x);
}
int max(int x, int y) {
	return (x>y ? x : y);
}

template <typename T>
vector<unsigned int> sort_indexes(const vector<T> &v) {

	// 初始化索引向量
	vector<unsigned int> idx(v.size());
	//使用iota对向量赋0~？的连续值
	iota(idx.begin(), idx.end(), 0);

	// 通过比较v的值对索引idx进行排序
	sort(idx.begin(), idx.end(),
		[&v](unsigned int i1, unsigned int i2) {return v[i1] < v[i2]; });
	return idx;
}

//int aa2idx[] = { 0, 2, 4, 3, 6, 13,7, 8, 9,20,11,10,12, 2,20,14,
//5, 1,15,16,20,19,17,20,18, 6 };
// idx for  A  B  C  D  E  F  G  H  I  J  K  L  M  N  O  P
//          Q  R  S  T  U  V  W  X  Y  Z
// so  aa2idx[ X - 'A'] => idx_of_X, eg aa2idx['A' - 'A'] => 0,
// and aa2idx['M'-'A'] => 12

int aa2idx_ACGT[] = { 0, -1, 1, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3 };
// index for          A   B  C   D   E   F  G   H   I   J   K   L   M   N   O   P   Q   R   S  T
// so  aa2idx_ACGT[ X - 'A'] => idx_of_X
// eg aa2idx_ACGT['A' - 'A'] => 0, aa2idx_ACGT['T' - 'A'] => 3. 

// kmer weight                  1  2   3   4   5     6      7      8      9      10
unsigned int NAAN_array[15] = { 1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144,
1048576, 4194304, 16777216, 67108864, 268435456 };
// 11       12        13        14         15


// node for sort
struct node
{
	unsigned int data;
	unsigned int index_in_genome;
	unsigned int index_in_sequence;
};

int comp(const void *a, const void *b) { // ascending order for struct
	return (*(struct node *)a).data> (*(struct node *)b).data ? 1 : -1;
}

struct TempFile
{
	FILE *file;
	char buf[512];

	TempFile(const char *dir = NULL) {
		int len = dir ? strlen(dir) : 0;
		assert(len < 400);
		buf[0] = 0;
		if (len) {
			strcat(buf, dir);
			if (buf[len - 1] != '/' && buf[len - 1] != '\\') {
				buf[len] = '/';
				len += 1;
			}
		}
		strcat(buf, "cdhit.temp.");
		len += 11;
		sprintf(buf + len, "%p", this);
		file = fopen(buf, "w+");
	}
	~TempFile() {
		if (file) {
			fclose(file);
			remove(buf);
		}
	}
};

struct TempFiles
{
	NVector<TempFile*> files;

	~TempFiles() { Clear(); }

	void Clear() {
		int i;
#pragma omp critical
		{
			for (i = 0; i<files.size; i++) if (files[i]) delete files[i];
			files.Clear();
		}
	}
};

const char *temp_dir = "";
TempFiles temp_files;

FILE* OpenTempFile(const char *dir = NULL)
{
	TempFile *file = new TempFile(dir);
#pragma omp critical
	{
		temp_files.files.Append(file);
	}
	return file->file;
}
static void CleanUpTempFiles()
{
	if (temp_files.files.Size()) printf("Clean up temporary files ...\n");
	temp_files.Clear();
}

void bomb_error(const char *message)
{
	fprintf(stderr, "\nFatal Error:\n%s\nProgram halted !!\n\n", message);
	temp_files.Clear();
	exit(1);
} // END void bomb_error

string getTime() {
	time_t timep;
	time(&timep);
	char tmp[64];
	strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S", localtime(&timep));
	return tmp;
}

// sort methods
// insert sorting and return the original index
// 插入排序法
vector<unsigned int> insertSort(vector<unsigned int> &num) {
	vector<unsigned int> index_original(num.size());
#pragma omp parallel for schedule( dynamic, 1 )
	for (unsigned int i = 1; i < num.size(); ++i) {
		index_original[i] = i;
	}
	for (unsigned int i = 1; i < num.size(); ++i) {
		for (unsigned int j = i; j > 0; --j) {
			if (num[j] < num[j - 1]) {
				unsigned int temp = num[j];
				num[j] = num[j - 1];
				num[j - 1] = temp;
				index_original[j] = j - 1;
				index_original[j - 1] = j;
			}
		}
	}
	return index_original;
}

void Merge(vector<unsigned int> & A, unsigned int left, unsigned int mid, unsigned int right)// 合并两个已排好序的数组A[left...mid]和A[mid+1...right]
{
	int len = right - left + 1;
	int *temp = new int[len];       // 辅助空间O(n)
	int index = 0;
	int i = left;                   // 前一数组的起始元素
	int j = mid + 1;                // 后一数组的起始元素
	while (i <= mid && j <= right)
	{
		temp[index++] = A[i] <= A[j] ? A[i++] : A[j++];  // 带等号保证归并排序的稳定性
	}
	while (i <= mid)
	{
		temp[index++] = A[i++];
	}
	while (j <= right)
	{
		temp[index++] = A[j++];
	}
	for (int k = 0; k < len; k++)
	{
		A[left++] = temp[k];
	}
	delete temp;
}

void MergeAndIndex(vector<unsigned int> & A, unsigned int left, unsigned int mid, unsigned int right, vector<unsigned int> & idx) {
	int len = right - left + 1;
	int *temp = new int[len];       // 辅助空间O(n)
	int *temp_index = new int[len];       // 辅助空间O(n)
	int index = 0;
	int i = left;                   // 前一数组的起始元素
	int j = mid + 1;                // 后一数组的起始元素
	while (i <= mid && j <= right)
	{
		if (A[i] <= A[j]) {
			temp[index] = A[i];
			temp_index[index++] = idx[i++];
		}
		else {
			temp[index] = A[j];
			temp_index[index++] = idx[j++];
		}
		//temp[index] = A[i] <= A[j] ? A[i++] : A[j++];  // 带等号保证归并排序的稳定性
		//temp_index[index++] = A[i] <= A[j] ? idx[i++] : idx[j++];
	}
	while (i <= mid)
	{
		temp[index] = A[i];
		temp_index[index++] = idx[i++];
	}
	while (j <= right)
	{
		temp[index] = A[j];
		temp_index[index++] = idx[j++];
	}
	for (int k = 0; k < len; k++)
	{
		A[left] = temp[k];
		idx[left++] = temp_index[k];
	}
	delete temp;
	delete temp_index;
}
void MergeAndIndex_with_matrix_point(unsigned int * A, unsigned int left, unsigned int mid, unsigned int right, unsigned int * idx) {
	unsigned int len = right - left + 1;
	unsigned int *temp = new unsigned int[len];       // 辅助空间O(n)
	unsigned int *temp_index = new unsigned int[len];       // 辅助空间O(n)
	unsigned int index = 0;
	unsigned int i = left;                   // 前一数组的起始元素
	unsigned int j = mid + 1;                // 后一数组的起始元素
	while (i <= mid && j <= right)
	{
		if (A[i] <= A[j]) {
			temp[index] = A[i];
			temp_index[index++] = idx[i++];
		}
		else {
			temp[index] = A[j];
			temp_index[index++] = idx[j++];
		}
		//temp[index] = A[i] <= A[j] ? A[i++] : A[j++];  // 带等号保证归并排序的稳定性
		//temp_index[index++] = A[i] <= A[j] ? idx[i++] : idx[j++];
	}
	while (i <= mid)
	{
		temp[index] = A[i];
		temp_index[index++] = idx[i++];
	}
	while (j <= right)
	{
		temp[index] = A[j];
		temp_index[index++] = idx[j++];
	}
	for (unsigned int k = 0; k < len; k++)
	{
		A[left] = temp[k];
		idx[left++] = temp_index[k];
	}
	delete temp;
	delete temp_index;
}

void MergeSortRecursion(vector<unsigned int> & A, unsigned int left, unsigned int right)    // 递归实现的归并排序(自顶向下)
{
	if (left == right)    // 当待排序的序列长度为1时，递归开始回溯，进行merge操作
		return;
	unsigned int mid = (left + right) / 2;
	MergeSortRecursion(A, left, mid);
	MergeSortRecursion(A, mid + 1, right);
	Merge(A, left, mid, right);
}
void MergeSortRecursionAndIndex(vector<unsigned int> & A, unsigned int left, unsigned int right, vector<unsigned int> & index)    // 递归实现的归并排序(自顶向下)
{
	if (left == right)    // 当待排序的序列长度为1时，递归开始回溯，进行merge操作
		return;
	unsigned int mid = (left + right) / 2;
	MergeSortRecursionAndIndex(A, left, mid, index);
	MergeSortRecursionAndIndex(A, mid + 1, right, index);
	MergeAndIndex(A, left, mid, right, index);
}
// sort matrix 
void MergeSortRecursionAndIndex_with_matrix_point(unsigned int * A, unsigned int left, unsigned int right, unsigned int * index)    // 递归实现的归并排序(自顶向下)
{
	if (left == right)    // 当待排序的序列长度为1时，递归开始回溯，进行merge操作
		return;
	unsigned int mid = (left + right) / 2;
	MergeSortRecursionAndIndex_with_matrix_point(A, left, mid, index);
	MergeSortRecursionAndIndex_with_matrix_point(A, mid + 1, right, index);
	MergeAndIndex_with_matrix_point(A, left, mid, right, index);
}

void MergeSortIterationAndIndex_with_matrix_point(unsigned int * A, unsigned int len, unsigned int * index) {
	long long int left, mid, right;
	for (long long int i = 1; i < len; i *= 2) {
		printf("i = %lld, len = %u\n", i, len);
		left = 0;
		while (left + i < len) {
			mid = left + i - 1;
			right = mid + i < len ? mid + i : len - 1;
			MergeAndIndex_with_matrix_point(A, left, mid, right, index);
			left = right + 1; 
		}
	}

}
void MergeSortIteration(vector<unsigned int> & A, unsigned int len)    // 非递归(迭代)实现的归并排序(自底向上)
{
	unsigned int left, mid, right;// 子数组索引,前一个为A[left...mid]，后一个子数组为A[mid+1...right]
	for (unsigned int i = 1; i < len; i *= 2)        // 子数组的大小i初始为1，每轮翻倍
	{
		left = 0;
		while (left + i < len)              // 后一个子数组存在(需要归并)
		{
			mid = left + i - 1;
			right = mid + i < len ? mid + i : len - 1;// 后一个子数组大小可能不够
			Merge(A, left, mid, right);
			left = right + 1;               // 前一个子数组索引向后移动
		}
	}
}




int print_usage() {
	cout << "*****************************************************************************" << endl;
	cout << "=============== Usage example(build genome library): ================" << endl;
	cout << "./smsAlign -build -i genome.fa -o genome_lib.txt" << endl << endl;
	cout << "-----build required:" << endl;
	cout << "-i      input genome fasta file (e.g. -i genome.fa)." << endl;
	cout << "-o      output library file (e.g. -o genome_lib.txt)." << endl;
	cout << " \n----build optional:" << endl;
	cout << "-k     k-mer length (default: 11)." << endl;


	cout << "\n=============== Uasge example(locate sequence position): ===============" << endl;
	cout << "./smsAlign -locate -i sequence.fasta -lib genomeLib.txt -o seqPos.txt" << endl;
	cout << "-----locate required:" << endl;
	cout << "-i      input sequence file (e.g. -i sequence.fa)." << endl;
	cout << "-lib    genome library file (e.g. -genome_lib.txt)." << endl;
	cout << "-o      output file (e.g. -o seqPos.txt)." << endl;
	cout << "\n----locate optional:" << endl;
	cout << "-T      the threads number (default: your computer has)." << endl;

	cout << "\n================ Uasge example(align sequences): ===============" << endl;
	cout << "./smsAlign -align -i sequence.fa -pos seqPos.txt -j genome.fa -o aligned.txt" << endl << endl;
	cout << "-----Align required:" << endl;
	cout << "-i      input sequence file (e.g. -i sequence.fa)." << endl;
	cout << "-j      genome sequence file (e.g. -j genome.fa)." << endl;
	cout << "-pos    sequence position file (e.g. -pos seqPos.txt)." << endl;
	cout << "\n----Align optional:" << endl;
	//cout << "-k     k-mer length (default: 11)." << endl;
	cout << "-T      the threads number (default: your computer has)." << endl;
	//cout << "-sam   SAM format output e.g. -sam samOut.txt (default: no sam output)." << endl;
	cout << "-ms     match score in the sequence alignment (default: 2)." << endl;
	cout << "-ss     substitution (mismatch) score in the sequence alignment (default: -2)." << endl;
	cout << "-gs     gap score in the sequence alignment (default: -2)." << endl;
	cout << "-h      print this usage (or --help)." << endl;
	return 0;

}

int print_usage_genome_library_build() {
	cout << "====================================================" << endl;
	cout << "Uasge example:" << endl;
	cout << "./smsAlign -build -i genome.fa -o genome_lib.txt" << endl << endl;
	cout << "-----build required:" << endl;
	cout << "-i      genome sequence file (e.g. -i genome.fa)." << endl;
	cout << "-o      output library file (e.g. -o genome_lib.txt)." << endl;
	cout << " \n----build optional:" << endl;
	cout << "-k      k-mer length (default: 15)." << endl;
	//cout << "Usage: ./smsAlignGenomeLibBuild genome_file output_file_name -k number of kmer (default 11), -T CPU number (default 10)" << endl;
	return 0;

}
int print_usage_sequence_position_locate() {
	cout << "====================================================" << endl;
	cout << "Usage example:" << endl;
	cout << "./smsAlign -locate -i sequence.fasta -lib genomeLib.txt -o seqPos.txt" << endl;
	cout << "-----Locate required:" << endl;
	cout << "-i      input sequence file (e.g. -i sequence.fa)." << endl;
	cout << "-lib    genome library file (e.g. -genome_lib.txt)." << endl;
	cout << "-o      output file (e.g. -o seqPos.txt)." << endl;
	cout << "\n----Locate optional:" << endl;
	cout << "-T      the threads number (default: your computer has)." << endl;
	return 0;
}
int print_usage_sequence_banded_alignment() {
	cout << "====================================================" << endl;
	cout << "Usage example:" << endl;
	cout << "smsMap --seq sequence.fa --pos pos.txt --genome genome.fa -out aligned.txt" << endl << endl;
	cout << "-----Align required:" << endl;
	cout << "--seq         input sequence file." << endl;
	cout << "--genome      genome sequence file." << endl;
	cout << "--pos         sequence position file." << endl;
	cout << "--out         output aligned file." << endl;
	//cout << "\n----Align optional:" << endl;
	//cout << "-k     k-mer length (default: 11)." << endl;
	//cout << "-T      the threads number (default: your computer has)." << endl;
	cout << "-h      print this usage (or --help)." << endl;
	//cout << "-sam   SAM format output e.g. -sam samOut.txt (default: no sam output)." << endl;
	//cout << "-ms     match score in the sequence alignment (default: 2)." << endl;
	//cout << "-ss     substitution (mismatch) score in the sequence alignment (default: -2)." << endl;
	//cout << "-gs     gap score in the sequence alignment (default: -2)." << endl;
	//cout << "-h      print this usage (or --help)." << endl;
	return 0;

}

bool Options::SetOptions(int argc, const char *argv[])
{
	int i, n;
	char date[100];
	strcpy(date, __DATE__);
	n = strlen(date);
	for (i = 1; i < n; i++) {
		if (date[i - 1] == ' ' && date[i] == ' ')
			date[i] = '0';
	}
	printf("================================================================\n");
	printf("Program: , V" VERSION WITH_OPENMP ", %s, " __TIME__ "\n", date);
	printf("Your command:");
	//n = 9;
	for (i = 0; i<argc; i++) {
		printf(" %s", argv[i]);
	}
	printf("\n\n");
	time_t tm = time(NULL);
	printf("Started: %s", ctime(&tm));
	printf("================================================================\n");
	printf("                            Output                              \n");
	printf("----------------------------------------------------------------\n");
	for (i = 1; i + 1<argc; i += 2) {
		//printf("\n argv[%d]= %s", i, argv[i]);
		//printf("\n argv[%d]= %s", i + 1, argv[i + 1]);
		bool dddd = SetOption(argv[i], argv[i + 1]);
		if (dddd == 0) return false;
	}
	if (i < argc) return false;

	atexit(CleanUpTempFiles);
	return true;
}

bool Options::SetOption(const char *flag, const char *value)
{
	bool ffff = SetOptionCommon(flag, value);
	return ffff;
}

void Options::SetKmerlength(int aaa) {
	kmer_length = aaa;
}

void Options::SetMaxBandedWidth(int ww) {
	max_banded_width = ww;
}

bool Options::SetOptionCommon(const char *flag, const char *value)
{
	int intval = atoi(value);
	if (strcmp(flag, "--seq") == 0) input = value;
	else if (strcmp(flag, "--genome") == 0) genome = value;
	else if (strcmp(flag, "-lib") == 0) genomeLibFile = value; //genome library file
	else if (strcmp(flag, "--pos") == 0) sequencePosFile = value; //genome library file
	else if (strcmp(flag, "--out") == 0) output = value;
	else if (strcmp(flag, "-m") == 0) model = value;
	else if (strcmp(flag, "-M") == 0) max_memory = atol(value) * 1000000;
	else if (strcmp(flag, "-k") == 0) kmer_length = intval;
	//else if (strcmp(flag, "-b") == 0) banded_width = intval;
	else if (strcmp(flag, "-p") == 0) print = intval;
	//else if (strcmp(flag, "-tmp") == 0) temp_dir = value;
	else if (strcmp(flag, "-min") == 0) min_length = intval;
	else if (strcmp(flag, "-ms") == 0) match_score = intval;
	else if (strcmp(flag, "-ss") == 0) mismatch_score = intval;
	else if (strcmp(flag, "-gs") == 0) gap_score = intval;
	else if (strcmp(flag, "-h") == 0 || strcmp(flag, "--help") == 0) {
		print_usage();
		exit(1);
	}
	else if (strcmp(flag, "-T") == 0) {
#ifndef NO_OPENMP
		int cpu = omp_get_num_procs();
		threads = intval;
		if (threads > cpu) {
			threads = cpu;
			printf("Warning: total number of CPUs in your system is %i\n", cpu);
		}
		if (threads == 0) {
			threads = cpu;
			printf("total number of CPUs in your system is %i\n", cpu);
		}
		if (threads != intval) printf("Actual number of CPUs to be used: %i\n\n", threads);
#else
		printf("Option -T is ignored: multi-threading with OpenMP is NOT enabled!\n");
#endif
	}
	else return false;
	return true;
}

void Options::Validate()
{
	if (model.compare("edlib") != 0 && model.compare("banded") != 0)
		bomb_error("Invalid alignment model (-m setting) detected!!!");
	if (kmer_length < 1 || kmer_length > 20) bomb_error("Invalid k-mer length, k should be 1 < k < 20");
	if (banded_width < 1) bomb_error("Invalid band width");
	if (min_length < 1) bomb_error("Invalid minimum length");
#ifndef NO_OPENMP
	int cpu = omp_get_num_procs();
	threads = cpu;
#else
	printf("Option -T is ignored: multi-threading with OpenMP is NOT enabled!\n");
#endif
}

void Options::Print()
{
	//printf("query_file = %i\n", input);
	printf("min_length = %i\n", min_length);
	printf("kmer_length = %i\n", kmer_length);
	printf("threads = %i\n", threads);
	printf("match_score = %i\n", match_score);
	printf("mismatch_score = %i\n", mismatch_score);
	printf("gap_score = %i\n", gap_score);
	printf("print = %i\n", print);
}

void Sequence::Clear()
{
	if (data) delete[] data;
	/* do not set size to zero here, it is need for writing output */
	bufsize = 0;
	data = NULL;
}
Sequence::Sequence()
{
	memset(this, 0, sizeof(Sequence));
	plus = -1;
	position_in_sequence = 0;
	position_in_genome = 0;
	position_in_genome_minus = 0;
	nMatch = 0;
	nSubsitute = 0;
	nDelete = 0;
	nInsert = 0;
}

Sequence::Sequence(const Sequence & other)
{
	int i;
	//printf( "new: %p  %p\n", this, & other );
	memcpy(this, &other, sizeof(Sequence));
	if (other.data) {
		size = bufsize = other.size;
		data = new char[size + 1];
		//printf( "data: %p  %p\n", data, other.data );
		data[size] = 0;
		memcpy(data, other.data, size);
		//for (i=0; i<size; i++) data[i] = other.data[i];
	}
	if (other.identifier) {
		int len = strlen(other.identifier);
		identifier = new char[len + 1];
		memcpy(identifier, other.identifier, len);
		identifier[len] = 0;
	}
}
Sequence::~Sequence()
{
	//printf( "delete: %p\n", this );
	if (data) delete[] data;
	if (identifier) delete[] identifier;
}

void Sequence::operator=(const char *s)
{
	size = 0; // avoid copying;
	Resize(strlen(s));
	strcpy(data, s);
}
void Sequence::operator+=(const char *s)
{
	int i, m = size, n = strlen(s);
	Reserve(m + n);
	memcpy(data + m, s, n);
}
void Sequence::Resize(int n)
{
	int i, m = size < n ? size : n;
	size = n;
	if (size != bufsize) {
		char *old = data;
		bufsize = size;
		data = new char[bufsize + 1];
		if (data == NULL) bomb_error("Memory");
		if (old) {
			memcpy(data, old, m);
			delete[]old;
		}
		if (size) data[size] = 0;
	}
}
void Sequence::Reserve(int n)
{
	int i, m = size < n ? size : n;
	size = n;
	if (size > bufsize) {
		char *old = data;
		bufsize = size + size / 5 + 1;
		data = new char[bufsize + 1];
		if (data == NULL) bomb_error("Memory");
		if (old) {
			memcpy(data, old, m);
			delete[]old;
		}
	}
	if (size) data[size] = 0;
}

void Sequence::Set_aligned_information(string a1, string a2, string midd, int nmattch, int nsubsitute, int ninsert, int deleten, int aligned_base_n, double sim) {
	aligned1 = new char[a1.size() + 1];
	aligned1[a1.size()] = 0;
	memcpy(aligned1, a1.c_str(), a1.size());

	aligned2 = new char[a2.size() + 1];
	aligned2[a2.size()] = 0;
	memcpy(aligned2, a2.c_str(), a2.size());

	middle = new char[midd.size() + 1];
	middle[midd.size()] = 0;
	memcpy(middle, midd.c_str(), midd.size());

	nMatch = nmattch;
	nSubsitute = nsubsitute;
	nDelete = deleten;
	nInsert = ninsert;
	identify = sim;
	nAlignedBase = aligned_base_n;
}

void Sequence::Set_sequence_locate(short direction, unsigned int pos_in_seq, unsigned int pos_in_gen, unsigned int pos_in_gen_minus) {
	plus = direction;
	position_in_sequence = pos_in_seq;
	if (direction == 1) {//plus direction
		position_in_genome = pos_in_gen;
	}
	else if (direction == 0) {
		position_in_genome_minus = pos_in_gen_minus;
	}
	else {
		position_in_genome = -1;
	}
}

/*
void Sequence::ConvertBases()
{
int i, indexBase;
//cout << *data << endl;
for (i = 0; i < size; i++) {
indexBase = aa2idx_ACGT[data[i] - 'A'];
data[i] = indexBase * weight;
//int a = toascii(data[i]);
//printf("haha=%d, toascii = %d\n", data[i], a);
//cout<<data[i];
}

}
*/
void Sequence::Swap(Sequence & other)
{
	Sequence tmp;
	memcpy(&tmp, this, sizeof(Sequence));
	memcpy(this, &other, sizeof(Sequence));
	memcpy(&other, &tmp, sizeof(Sequence));
	memset(&tmp, 0, sizeof(Sequence));
}
int Sequence::Format()
{
	int i, j = 0, m = 0;
	while (size && isspace(data[size - 1])) size--;
	if (size && data[size - 1] == '*') size--;
	if (size) data[size] = 0;
	for (i = 0; i<size; i++) {
		char ch = data[i];
		m += !(isalpha(ch) | isspace(ch));
	}
	if (m) return m;
	for (i = 0; i<size; i++) {
		char ch = data[i];
		if (isalpha(ch)) data[j++] = toupper(ch);
	}
	data[j] = 0;
	size = j;
	return 0;
}

void Sequence::SwapIn()
{
	if (data) return;
	if (swap == NULL) bomb_error("Can not swap in sequence");
	Resize(size);
	fseek(swap, offset, SEEK_SET);
	if (fread(data, 1, size, swap) == 0) bomb_error("Can not swap in sequence");
	data[size] = 0;
}
void Sequence::SwapOut()
{
	if (swap && data) {
		delete[] data;
		bufsize = 0;
		data = NULL;
	}
}

void SequenceDB::Read(const char *file, const Options & options)
{
	Sequence one;
	Sequence dummy;
	Sequence des;
	Sequence *last = NULL;
	FILE *swap = NULL;
	FILE *fin = fopen(file, "r");
	char *buffer = NULL;
	char *res = NULL;
	size_t swap_size = 0;
	int option_l = options.min_length;
	if (fin == NULL) bomb_error("Failed to open the sequence file");
	Clear();
	dummy.swap = swap;
	buffer = new char[MAX_LINE_SIZE + 1];

	while (!feof(fin) || one.size) { /* do not break when the last sequence is not handled */
		buffer[0] = '>';
		if ((res = fgets(buffer, MAX_LINE_SIZE, fin)) == NULL && one.size == 0)
			break;
		if (buffer[0] == '+') {
			int len = strlen(buffer);
			int len2 = len;
			while (len2 && buffer[len2 - 1] != '\n') {
				if ((res = fgets(buffer, MAX_LINE_SIZE, fin)) == NULL) break;
				len2 = strlen(buffer);
				len += len2;
			}
			one.des_length2 = len;
			dummy.des_length2 = len;
			fseek(fin, one.size, SEEK_CUR);
		}
		else if (buffer[0] == '>' || buffer[0] == '@' || (res == NULL && one.size)) {
			if (one.size) { // write previous record
				one.dat_length = dummy.dat_length = one.size;
				if (one.identifier == NULL || one.Format()) {
					printf("Warning: from file \"%s\",\n", file);
					printf("Discarding invalid sequence or sequence without header and description!\n\n");
					if (one.identifier) printf("%s\n", one.identifier);
					printf("%s\n", one.data);
					one.size = 0;
				}
				one.index = dummy.index = sequences.size();
				if (one.size > 0) {
					if (swap) {
						swap_size += one.size;
						// so that size of file < MAX_BIN_SWAP about 2GB
						if (swap_size >= MAX_BIN_SWAP) {
							dummy.swap = swap = OpenTempFile(temp_dir);
							swap_size = one.size;
						}
						dummy.size = one.size;
						dummy.offset = ftell(swap);
						dummy.des_length = one.des_length;
						dummy.plus = -1;
						sequences.Append(new Sequence(dummy));
						//one.ConvertBases();
						fwrite(one.data, 1, one.size, swap);
					}
					else {
						//printf( "==================\n" );
						sequences.Append(new Sequence(one));
						//printf( "------------------\n" );
						//if( sequences.size() > 10 ) break;
					}
					//if( sequences.size() >= 10000 ) break;
				}
			}
			one.size = 0;
			one.des_length2 = 0;

			int len = strlen(buffer);
			int len2 = len;
			des.size = 0;
			des += buffer;
			while (len2 && buffer[len2 - 1] != '\n') {
				if ((res = fgets(buffer, MAX_LINE_SIZE, fin)) == NULL) break;
				des += buffer;
				len2 = strlen(buffer);
				len += len2;
			}
			size_t offset = ftell(fin);
			one.des_begin = dummy.des_begin = offset - len;
			one.des_length = dummy.des_length = len;

			int i = 0;
			if (des.data[i] == '>' || des.data[i] == '@' || des.data[i] == '+')
				i += 1;
			if (des.data[i] == ' ' || des.data[i] == '\t')
				i += 1;
			//if (options.des_len && options.des_len < des.size) des.size = options.des_len;
			while (i < des.size && !isspace(des.data[i]))
				i += 1;
			des.data[i] = 0;
			one.identifier = dummy.identifier = des.data;
		}
		else {
			one += buffer;
		}
	}
#if 0
	int i, n = 0;
	for (i = 0; i<sequences.size(); i++) n += sequences[i].bufsize + 4;
	cout << n << "\t" << sequences.capacity() * sizeof(Sequence) << endl;
	int i;
	scanf("%i", &i);
#endif
	one.identifier = dummy.identifier = NULL;
	delete[] buffer;
	fclose(fin);
}
void SequenceDB::SequenceStatistic(Options & options)
{
	int i, j, k, len;
	int N = sequences.size();
	total_letter = 0; // total bases
	total_desc = 0;   // total letters of sequence headers
	max_len = 0;
	mean_len = 0.0;
	min_len = (size_t)-1;
	for (i = 0; i<N; i++) {
		Sequence *seq = sequences[i];
		len = seq->size;
		mean_len += seq->size;
		total_letter += len;
		if (len > max_len)
			max_len = len;
		if (len < min_len)
			min_len = len;
		if (seq->swap == NULL)
			//			seq->ConvertBases();
			if (seq->identifier)
				total_desc += strlen(seq->identifier);
	}
	mean_len = mean_len / N;
	cout << "  Total number: " << N << endl;
	cout << "  Longest:      " << max_len << endl;
	cout << "  Shortest:     " << min_len << endl;
	cout << "  Mean length:  " << mean_len << endl;
	cout << "  Total bases:  " << total_letter << endl;
	cout << endl;
	// END change all the NR_seq to iseq

	//len_n50 = (max_len + min_len) / 2; // will be properly set, if sort is true;

}// END sort_seqs_divide_segs

void SequenceDB::SequencePositionWriteToFile(Options & options) {
	string time_now = getTime();
	ofstream outfile;
	const char* filename = options.output.data();
	outfile.open(filename, ios::out);
	//outfile2.open(options.output, ios::app);
	if (!outfile.is_open())
		bomb_error("Open sequence positiony output file failure, exit!");
	//for (int aaa = 0; aaa < location_num; aaa++) {
	outfile << "Build date:   " << time_now << endl;
	outfile << "Word length:  " << options.kmer_length << endl;
	outfile << "Sequence num: " << sequences.size() << endl;
	outfile << "Max. length:  " << max_len << endl;
	outfile << "Min. length:  " << min_len << endl;
	outfile << "Ave. length:  " << mean_len << endl;
	outfile << "SeqIndex\tPlus\tPosInSeq\tPosInGen" << endl;
	for (int i = 0; i < sequences.size(); i++) {
		outfile << i << "\t";
		outfile << sequences[i]->plus << "\t";
		//if (i > 100) {
		//	continue;
		//}
		//printf("%d-th in sequence: %d, in genome: %d\n", i, sequences[i]->position_in_sequence, sequences[i]->position_in_genome);
		if (sequences[i]->plus == 1) {
			outfile << sequences[i]->position_in_sequence << "\t";
			outfile << sequences[i]->position_in_genome;
		}
		else if (sequences[i]->plus == 0) {
			outfile << sequences[i]->position_in_sequence << "\t";
			outfile << sequences[i]->position_in_genome_minus;
		}
		else {
			outfile << "-1" << "\t";
			outfile << "-1";
		}
		outfile << endl;
	}
	outfile.close();
}

void SequenceDB::ReadSequencePositionFile(Options & options) {
	const char* filename = options.sequencePosFile.data();
	ifstream libfile(filename);
	string line, kmer, w1, w2;
	int index, plus, genome_id, positionInSequence, positionInGenome, kmer_length;
	if (!libfile.is_open())
		bomb_error("Open genome library output file failure, exit!");
	// ignore the first four lines
	//getline(libfile, line);// data
	//getline(libfile, line);// word length
	//istringstream iss(line);
	//iss >> w1 >> w2 >> kmer_length;
	//getline(libfile, line);// sequence num
	//getline(libfile, line);// Max length
	//getline(libfile, line);// Min length
	//getline(libfile, line);// Average length
	//getline(libfile, line);// title
	while (getline(libfile, line)) {
		istringstream iss(line);
		iss >> index >> genome_id >> plus >> positionInSequence >> positionInGenome;
		if (plus == 1) {
			sequences[index]->plus = 1;
			sequences[index]->tar_id = genome_id;
			sequences[index]->position_in_sequence = positionInSequence;
			sequences[index]->position_in_genome = positionInGenome;
		}
		else if (plus == 0) {
			sequences[index]->plus = 0;
			sequences[index]->tar_id = genome_id;
			sequences[index]->position_in_sequence = positionInSequence;
			sequences[index]->position_in_genome_minus = positionInGenome;
		}
		else if (plus == -1) {
			sequences[index]->plus = -1;
			sequences[index]->tar_id = genome_id;
			sequences[index]->position_in_sequence = -1;
			sequences[index]->position_in_genome = -1;
		}
		else {
			printf("\nInvalid position detected in the %d-th sequence!\n", index);
			bomb_error("Invalid position detected in the sequence position file, exit!");
		}
	}
}

void SequenceDB::WriteAlignedToFile(Options & options) {
	ofstream outfile2;
	string direction = "";
	string seq1_whole_aligned = "";
	string middle_whole_aligned = "";
	string seq2_whole_aligned = "";
	int leftPosition = 0, rightPosition = 0;
	const char* filename = options.output.data();
	outfile2.open(filename, ios::out);
	//outfile2.open(options.output, ios::app);
	if (!outfile2.is_open())
		bomb_error("Open output file failure");
	for (int i = 0; i < sequences.size(); i++) {
		if (sequences[i]->nMatch <= 0) {
			continue;
		}
		if (sequences[i]->plus) {
			direction = "Plus";
		}
		else {
			direction = "Minus";
		}
		outfile2 << "Query:        " << sequences[i]->identifier << endl;
		outfile2 << "Length:       " << sequences[i]->size << endl;
		outfile2 << "nMatch:       " << sequences[i]->nMatch << endl;
		outfile2 << "nSubsitute:   " << sequences[i]->nSubsitute << endl;
		outfile2 << "nDelete:      " << sequences[i]->nDelete << endl;
		outfile2 << "nInsert:      " << sequences[i]->nInsert << endl;
		outfile2 << "Identify:     " << sequences[i]->identify << endl;
		outfile2 << "Strand:       Plus/" << direction << endl;
		outfile2 << "nAlignedBase: " << sequences[i]->nAlignedBase << endl;
		outfile2 << endl;
		leftPosition = 0;
		rightPosition = 0;
		seq1_whole_aligned = sequences[i]->aligned1;
		middle_whole_aligned = sequences[i]->middle;
		seq2_whole_aligned = sequences[i]->aligned2;
		for (int iii = 0; iii < seq1_whole_aligned.size(); ) {
			if (leftPosition >= seq1_whole_aligned.size()) {
				break;
			}
			outfile2 << "Query  " << seq1_whole_aligned.substr(leftPosition, 60) << endl;
			outfile2 << "       " << middle_whole_aligned.substr(leftPosition, 60) << endl;
			outfile2 << "Sbjct  " << seq2_whole_aligned.substr(leftPosition, 60) << endl;
			outfile2 << endl;
			leftPosition += 60;
		}
		outfile2 << endl;
		outfile2 << endl;
	}

	outfile2.close();
}

void GenomeDB::Read_my(const char *file, const Options & options) {
	//genome[0] = new Sequence();
	//genome->data = new char[2861343702];
	//Sequence one;
	//genome.Append(new Sequence(one));
	//Sequence one;
	Vector<unsigned long long int> pos_enter;
	char * data = NULL;
	//string sequence_is = "";
	const char* filename = file;// options.genomeLibFile.data();
	string header = "";
	string sequence = "";
	unsigned int seq_len = 0;
	unsigned int len = 0;
	int flag = 1; // decide the direction in the reading procedure
				  //libfile.open(filename);
				  //outfile2.open(options.output, ios::app);
	int fd = open(filename, O_RDONLY);
	int firstt = 0;
	int num_genome = 0, pass_enter;
	unsigned int size = lseek(fd, 0, SEEK_END);
	data = (char *)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
	close(fd);
	int flag1, flag2, flag_head, seq_num = -1;
	printf("\n");
	for (unsigned int i = 0; i < size; i++) {
		if (data[i] == '>') {
			flag_head = 1;
			seq_num += 1;
			flag1 = 1;
			pass_enter = 0;
			if (seq_num > 0){
				if (len > 0) {
					printf("Genome name: %s\t length: %d\n", header.data(), len);
					genome[seq_num - 1].data = new char [len + 1];
					genome[seq_num - 1].size = len;
					genome[seq_num - 1].header = header;
					genome_minus_sets[seq_num - 1].data = new char [len + 1];
                                        genome_minus_sets[seq_num - 1].size = len;
                                        genome_minus_sets[seq_num - 1].header = header;
					transform(sequence.begin(), sequence.end(), sequence.begin(), ::toupper);
					//transform(wstr.begin(), wstr.end(), wstr.begin(), towupper);
					strcpy(genome[seq_num - 1].data,sequence.c_str());
					reverse(sequence.begin(), sequence.end());
					for (unsigned int u = 0; u < len; u++){
						switch (sequence[u]) {
							case 'A':
								genome_minus_sets[seq_num - 1].data[u] = 'T';
								break;
							case 'C':
                                                                genome_minus_sets[seq_num - 1].data[u] = 'G';
                                                                break;
							case 'G':
                                                                genome_minus_sets[seq_num - 1].data[u] = 'C';
                                                                break;
							case 'T':
                                                                genome_minus_sets[seq_num - 1].data[u] = 'A';
                                                                break;
							default:
								genome_minus_sets[seq_num - 1].data[u] = sequence[u];
						}
					}
					genome_minus_sets[seq_num - 1].data[len] = '\0';
				}
				sequence = "";
				flag_head = -1;	
				len = 0;
				header = "";
			}
		continue;	
		}
		if (data[i] == '\n') {
			pass_enter += 1;
		}
		if (pass_enter >= 1) {
			if (data[i] == '\n'){
				continue;
			}
			else {
				sequence += data[i];
				len++;
			}
		}
		else {
			if (data[i] != '\n'){
				header += data[i];
			}
		}
	}
	if (len > 0) {
		printf("Genome name: %s\t length: %d\n", header.data(), len);
		genome[seq_num].data = new char [len + 1];
		genome[seq_num].size = len;
		genome[seq_num].header = header;
		genome_minus_sets[seq_num].data = new char [len + 1];
                genome_minus_sets[seq_num].size = len;
                genome_minus_sets[seq_num].header = header;
		transform(sequence.begin(), sequence.end(), sequence.begin(), ::toupper);
		strcpy(genome[seq_num].data,sequence.c_str());
		reverse(sequence.begin(), sequence.end());
		for (unsigned int u = 0; u < len; u++){
			switch (sequence[u]) {
                        	case 'A':
                                	genome_minus_sets[seq_num].data[u] = 'T';
                                        break;
                        	case 'C':
                                	genome_minus_sets[seq_num].data[u] = 'G';
                                        break;
              			case 'G':
                                	genome_minus_sets[seq_num].data[u] = 'C';
                                        break;
               	                case 'T':
                                        genome_minus_sets[seq_num].data[u] = 'A';
                                        break;
                		default:
                                	genome_minus_sets[seq_num].data[u] = sequence[u];
			}
		}
		genome_minus_sets[seq_num].data[len] = '\0';		
        }
	genome_num = seq_num + 1;
	//cout << header << endl;
	unsigned int bases = 0;
	printf("\n");
	for (int y = 0; y < genome_num; y++){
		//printf("Genome name: %s\t length: %d\n", genome[y].header.data(), genome[y].size);
		bases +=  genome[y].size;
	}
	printf("Total number: %d, total bases: %u\n", genome_num, bases);
	//printf("Genome added, genome length: %lu\n", sequence.length());
}



void GenomeDB::GenomeMinus() {
	//Sequence seq = genome[0];
	//Sequence genome_mimus = genome_minus_sets[0];
	//genome_minus_sets[0].size = seq.size;
	//genome_minus_sets[0].data = new char[seq.size];//
	unsigned int len;
	unsigned int j = 0;
	//for (int i = 0; i < genome_num; i++){
	//	len = genome[i].size;
	//	genome_minus_sets[i].data = new char [len + 1];
	//	genome_minus_sets[i].size = len;
	//	j = 0;
	//	for (int k = len - 1; k >=0; k--){
	//		switch (genome[i].data[k]){
	//			case 'A':
	//				genome_minus_sets[i].data[j] = 'T';
	//				break;
	//			case 'C':
	//				genome_minus_sets[i].data[j] = 'G';
        //                              break;
	//			case 'G':
	//				genome_minus_sets[i].data[j] = 'C';
        //                              break;
	//			case 'T':
	//				genome_minus_sets[i].data[j] = 'A';
	//				break;
	//			//default:
	//			//	genome_minus_sets[i].data[j] = genome[i].data[j];
	//		}
	//		j++;	
	//	}
	//	genome_minus_sets[i].data[len] = '\0';
	//	printf("Reversing %d done, total %d\n", i + 1, genome_num);
	//}
}
void GenomeDB::GenomeIndexBuildForMulti_threads(unsigned int i, const Options & options) {
	unsigned int weight, temp, encode = 0;
	int k2 = options.kmer_length;
	int k_len = options.kmer_length;
	//for (int j = 0; j < k_len; j++) {
	//	k2--;
	//	weight = aa2idx_ACGT[genome[0]->data[i + j] - 'A'];
	//	temp = weight * NAAN_array[k2];
	//	encode += temp;
	//kmerP += seq->data[i + j];
	//kmerM += genome_minus->data[i + j];
	//}
	//hash_value_each_position[i] = encode;
}
void GenomeDB::GenomeIndexBuildAndLocate(SequenceDB & seqsDB, const Options & options) {
	Sequence *seq = genome; // this version we just use one genome
							   //genome.Append(new Sequence(genome[0]));
	genome_minus = new Sequence();
	genome_minus->size = seq->size;
	genome_minus->data = new char[seq->size];//
	printf("\nGenome length: %d\n\n", seq->size);
	int ii = -1;
	for (int mm = seq->size - 1; mm >= 0; mm--) {
		ii++;
		switch (seq->data[mm]) {
		case 'A':
			genome_minus->data[ii] = 'T';
			break;
		case 'C':
			genome_minus->data[ii] = 'G';
			break;
		case 'G':
			genome_minus->data[ii] = 'C';
			break;
		case 'T':
			genome_minus->data[ii] = 'A';
			break;
		}
	}
	unsigned int seq_len = seq->size;
	int k_len = options.kmer_length;
	unsigned int hash_list_size = seq_len - k_len + 1;
	//hash_list_size = 10000000;
	float p = 0.0, p0 = 0.0;
	int k2 = options.kmer_length; // the digit index
	int unique_kmer = -1;
	unsigned int i, j, weight, weight2, hash_value_is, hash_value_is2, hash_value_temp;
	unsigned int temp, temp2, encode, encode2; // unsigned int:  0～4294967295
	string time2, time1 = getTime();
	unsigned int all_kmer_nums = pow(4, options.kmer_length);
	hash_value_each_position_point = new unsigned int[hash_list_size];
	//unsigned int * hash_value_original = new unsigned int[hash_list_size];
	hash_value_index = new unsigned int[hash_list_size];// index in the genome
														// store the hash value index in the sorted hash_value_each_position_point
														// and not exist: -1 
	kmer_exist_and_index = new long long int[all_kmer_nums];
	kmer_exist_and_index_minus = new long long int[all_kmer_nums];
	for (i = 0; i < all_kmer_nums; i++) {
		kmer_exist_and_index[i] = -1;
		kmer_exist_and_index_minus[i] = -1;
	}
	for (i = 0; i < hash_list_size; i++) {
		k2 = k_len;
		encode = 0;
		for (int j = 0; j < k_len; j++) {
			k2--;
			weight = aa2idx_ACGT[genome->data[i + j] - 'A'];
			temp = weight * NAAN_array[k2];
			encode += temp;
		}
		hash_value_each_position_point[i] = encode;
		hash_value_index[i] = i;
		//if (i >= 50)
		//	break;
		//GenomeIndexBuildForMulti_threads(i, options);
		//if (omp_get_thread_num() == 1) {
		//	p = (100.0*i) / hash_list_size;
		//	if (p > p0 + 1E-1) {
		//		printf("\r%5.1f%%   %10d-th, total %10d", p, i, hash_list_size); //printf("The %d-th sequence is located", i);
		//		p0 = p;
		//	}
		//}
		//fflush(stdout);
	}
	time1 = getTime();
	// Merge Sort Recursion model for the matrix (not vector)
	MergeSortRecursionAndIndex_with_matrix_point(hash_value_each_position_point, 0, hash_list_size - 1, hash_value_index);
	time2 = getTime();
	cout << "\n\n-------Merge Sorting Recursion and sort index with matrix:" << endl;
	cout << "Start at: " << time1 << endl;
	cout << "End   at: " << time2 << endl;
	hash_value_is = hash_value_each_position_point[0];
	kmer_exist_and_index[hash_value_is] = 0;
	hash_value_temp = hash_value_is;
	for (i = 1; i < hash_list_size; i++) {
		hash_value_is2 = hash_value_each_position_point[i];
		if (hash_value_is2 == hash_value_temp)
			continue;
		kmer_exist_and_index[hash_value_is2] = i;
		hash_value_temp = hash_value_is2;
		//printf("kmer_exist_and_index[%d] = %lld\n", hash_value_is2, kmer_exist_and_index[hash_value_is2]);
	}
	printf("    Plus direction done.\n\n");
	time1 = getTime();
	printf("Building minus direction...");
	hash_value_each_position_point_minus = new unsigned int[hash_list_size];
	//unsigned int * hash_value_original = new unsigned int[hash_list_size];
	hash_value_index_minus = new unsigned int[hash_list_size];// index in the genome

															  //genomeIndexLocateMinus.resize(pow(4, options.kmer_length) + 1);
	for (i = 0; i < seq_len - k_len + 1; i++) {
		encode2 = 0;
		k2 = options.kmer_length;
		for (j = 0; j < k_len; j++) {
			k2--;
			weight2 = aa2idx_ACGT[genome_minus->data[i + j] - 'A'];
			temp2 = weight2 * NAAN_array[k2];
			encode2 += temp2;
		}
		hash_value_each_position_point_minus[i] = encode2;
		hash_value_index_minus[i] = i;
	}
	MergeSortRecursionAndIndex_with_matrix_point(hash_value_each_position_point_minus, 0, hash_list_size - 1, hash_value_index_minus);
	hash_value_is = hash_value_each_position_point_minus[0];
	kmer_exist_and_index_minus[hash_value_is] = 0;// hash_value_index_minus[0];
	hash_value_temp = hash_value_is;
	for (i = 1; i < hash_list_size; i++) {
		hash_value_is2 = hash_value_each_position_point_minus[i];
		if (hash_value_is2 == hash_value_temp)
			continue;
		kmer_exist_and_index_minus[hash_value_is2] = i;// hash_value_index_minus[i];
		hash_value_temp = hash_value_is2;
	}
	time2 = getTime();
	cout << "\n\n----- Minus direction build:" << endl;
	cout << "Start at: " << time1 << endl;
	cout << "End   at: " << time2 << endl;
	printf("    Done.\n\n");

	printf("Locating sequence position...");

	int tid = 0;
	//#pragma omp parallel for schedule( dynamic, 1 )
	for (i = 0; i < seqsDB.sequences.size(); i++) {
		if (seqsDB.sequences[i]->size <= 10000) {
			//int aaaa = 0;
			//continue;
		}
		//for (int ii = 883101; ii < 883212; ii++) {
		//	printf("kmer_exist_and_index[%d] = %lld\n", ii, kmer_exist_and_index[ii]);
		//}
		SequenceLocateInGenome_merge_sort(seqsDB.sequences[i], options);//, 
																		//hash_value_each_position_point, hash_value_index, kmer_exist_and_index, 
																		//hash_value_each_position_point_minus, hash_value_index_minus, kmer_exist_and_index_minus); // find the homologous region																		 //printf("%d-th in sequence: %d, in genome: %d\n", i, seq_db.sequences[i]->position_in_sequence, seq_db.sequences[i]->position_in_genome);																//printf("The %d-th sequence position is located\n", i);
		break;
		tid = omp_get_thread_num();
		if (omp_get_thread_num() == 0) {
			p = (100.0*i) / seqsDB.sequences.size();
			if (p > p0 + 1E-1) {
				printf("\r%4.1f%%   %8d-th sequence", p, i); //printf("The %d-th sequence is located", i);
				p0 = p;
			}
		}
		fflush(stdout);
	}


	//printf("kmer_exist_and_index size = %ld\n", sizeof(kmer_exist_and_index)/sizeof(kmer_exist_and_index[0]));
}


void GenomeDB::GenomeIndexBuildWriteToFile(const Options & options) {
	Sequence *seq = genome; // this version we just use one genome
							   //genome.Append(new Sequence(genome[0]));
	genome_minus = new Sequence();
	genome_minus->size = seq->size;
	genome_minus->data = new char[seq->size];//
	printf("\n    Genome length: %u\n", seq->size);
	unsigned int ii = -1;
	for (long long int mm = seq->size - 1; mm >= 0; mm--) {
		ii++;
		switch (seq->data[mm]) {
		case 'A':
			genome_minus->data[ii] = 'T';
			break;
		case 'C':
			genome_minus->data[ii] = 'G';
			break;
		case 'G':
			genome_minus->data[ii] = 'C';
			break;
		case 'T':
			genome_minus->data[ii] = 'A';
			break;
		}
	}
	unsigned int seq_len = seq->size;
	int k_len = options.kmer_length;
	unsigned int hash_list_size = seq_len - k_len + 1;
	//hash_list_size = 10000000;
	float p = 0.0, p0 = 0.0;
	int k2 = options.kmer_length; // the digit index
	int unique_kmer = -1;
	unsigned int i, j, weight, weight2, hash_value_is, hash_value_is2, hash_value_temp;
	unsigned int temp, temp2, encode, encode2; // unsigned int:  0～4294967295
	string time2, time1 = getTime();
	unsigned int all_kmer_nums = pow(4, options.kmer_length);
	unsigned int maxx_encode;
	hash_value_each_position_point = new unsigned int[hash_list_size];
	hash_value_index = new unsigned int[hash_list_size];// index in the genome
														// store the hash value index in the sorted hash_value_each_position_point
														// and not exist: -1 
	printf("    Building plus direction...");
	//kmer_exist_and_index = new long long int[all_kmer_nums];
	//for (i = 0; i < all_kmer_nums; i++) {
	//	kmer_exist_and_index[i] = -1;
	//}
	string ss = "";
	for (i = 0; i < hash_list_size; i++) {
		k2 = k_len;
		encode = 0;
		ss = "";
		for (int j = 0; j < k_len; j++) {
			k2--;
			weight = aa2idx_ACGT[genome->data[i + j] - 'A'];
			temp = weight * NAAN_array[k2];
			encode += temp;
			ss += genome->data[i + j];
		}
		//if (encode > 1073741823) {
		//	printf("Position: %d, k-mer: %s\n, encode: %d\n",i, ss, encode);
		//}
		hash_value_each_position_point[i] = encode;
		hash_value_index[i] = i;
	}
	// Merge Sort Recursion model for the matrix (not vector)
	//MergeSortRecursionAndIndex_with_matrix_point(hash_value_each_position_point, 0, hash_list_size - 1, hash_value_index);
	MergeSortIterationAndIndex_with_matrix_point(hash_value_each_position_point, hash_list_size, hash_value_index);
	//hash_value_is = hash_value_each_position_point[0];
	//kmer_exist_and_index[hash_value_is] = hash_value_index[0];
	//hash_value_temp = hash_value_is;
	//for (i = 1; i < hash_list_size; i++) {
	//	hash_value_is2 = hash_value_each_position_point[i];
	//	if (hash_value_is2 == hash_value_temp)
	//		continue;
	//	kmer_exist_and_index[hash_value_is2] = hash_value_index[i];
	//	hash_value_temp = hash_value_is2;
	//printf("kmer_exist_and_index[%d] = %lld\n", hash_value_is2, kmer_exist_and_index[hash_value_is2]);
	//}
	printf("    Done.\n");
	time1 = getTime();
	printf("    Building minus direction...");
	hash_value_each_position_point_minus = new unsigned int[hash_list_size];
	//unsigned int * hash_value_original = new unsigned int[hash_list_size];
	hash_value_index_minus = new unsigned int[hash_list_size];// index in the genome
															  //kmer_exist_and_index_minus = new long long int[all_kmer_nums];
															  //genomeIndexLocateMinus.resize(pow(4, options.kmer_length) + 1);
	for (i = 0; i < seq_len - k_len + 1; i++) {
		encode2 = 0;
		k2 = options.kmer_length;
		for (j = 0; j < k_len; j++) {
			k2--;
			weight2 = aa2idx_ACGT[genome_minus->data[i + j] - 'A'];
			temp2 = weight2 * NAAN_array[k2];
			encode2 += temp2;
		}
		hash_value_each_position_point_minus[i] = encode2;
		hash_value_index_minus[i] = i;
	}
	//MergeSortIterationAndIndex_with_matrix_point(hash_value_each_position_point_minus, hash_list_size, hash_value_index_minus);
	//MergeSortRecursionAndIndex_with_matrix_point(hash_value_each_position_point_minus, 0, hash_list_size - 1, hash_value_index_minus);
	//hash_value_is = hash_value_each_position_point_minus[0];
	//kmer_exist_and_index_minus[hash_value_is] = hash_value_index_minus[0];
	//hash_value_temp = hash_value_is;
	//for (i = 1; i < hash_list_size; i++) {
	//	hash_value_is2 = hash_value_each_position_point_minus[i];
	//	if (hash_value_is2 == hash_value_temp)
	//		continue;
	//	kmer_exist_and_index_minus[hash_value_is2] = hash_value_index_minus[i];
	//	hash_value_temp = hash_value_is2;
	//}
	//time2 = getTime();
	//cout << "\n\n----- Minus direction build:" << endl;
	//cout << "Start at: " << time1 << endl;
	//cout << "End   at: " << time2 << endl;
	//printf("    Done.\n\n");
	printf("    Done.\n");
	printf("    Writing plus direction genome library...");
	string time_now = getTime();
	ofstream outfile;
	const char* filename = options.output.data();
	outfile.open(filename, ios::out);
	//outfile2.open(options.output, ios::app);
	if (!outfile.is_open())
		bomb_error("Open genome library output file failure, exit!");
	//for (int aaa = 0; aaa < location_num; aaa++) {
	outfile << "//Build date:  " << time_now << endl;
	outfile << "//Genome length:  " << seq_len << endl;
	outfile << "//K-mer length: " << options.kmer_length << endl;
	outfile << "//Direction:   Plus" << endl;
	outfile << "//Hash encode sorted\tPositions in genome" << endl;
	for (unsigned int kkk = 0; kkk < hash_list_size; kkk++) {
		outfile << hash_value_each_position_point[kkk] << "\t"; // unique kmer index
		outfile << hash_value_index[kkk] << endl; // position in the genome
	}
	printf("    Done.\n");
	printf("    Writing minus direction genome library...");
	delete[] hash_value_each_position_point;
	delete[] hash_value_index;
	MergeSortIterationAndIndex_with_matrix_point(hash_value_each_position_point_minus, hash_list_size, hash_value_index_minus);
	outfile << "//Direction  : Minus\t" << endl;
	outfile << "//Hash encode sorted\tPositions in genome (Minus)" << endl;
	for (unsigned int kkk = 0; kkk < hash_list_size; kkk++) {
		outfile << hash_value_each_position_point_minus[kkk] << "\t"; // unique kmer index
		outfile << hash_value_index_minus[kkk] << endl; // position in the genome
	}
	printf("    Done.\n\n");
	outfile.close();
	//delete[] hash_value_each_position_point;
	//delete[] hash_value_index;
	delete[] hash_value_each_position_point_minus;
	delete[] hash_value_index_minus;
}
int GenomeDB::GenomeLibraryRead(const Options & options) {
	const char* filename = options.genomeLibFile.data();
	string line, kmer, w1, w2;
	unsigned int positionsInGenome, encode, a, kmer_length, pp;
	char *s_line, *p;
	const char * split = ",";
	unsigned int seq_len, hash_value_is, hash_value_temp, hash_value_is2;
	int flag = 1; // decide the direction in the reading procedure
				  //libfile.open(filename);
				  //outfile2.open(options.output, ios::app);
	ifstream libfile(filename);
	if (!libfile.is_open())
		bomb_error("Open genome library output file failure, exit!");
	// ignore the first four lines
	getline(libfile, line);// data
	getline(libfile, line);// genome length
	istringstream iss(line);
	iss >> w1 >> w2 >> seq_len;
	getline(libfile, line);// word length
	istringstream iss2(line);
	iss2 >> w1 >> w2 >> kmer_length;
	getline(libfile, line);// direction
	getline(libfile, line);// title
	unsigned int all_kmer_nums = pow(4, kmer_length);
	unsigned int hash_list_size = seq_len - kmer_length + 1;
	hash_value_each_position_point = new unsigned int[hash_list_size];
	hash_value_index = new unsigned int[hash_list_size];// index in the genome
	kmer_exist_and_index = new long long int[all_kmer_nums]; // store the hash value index in the sorted hash_value_each_position_point
	hash_value_each_position_point_minus = new unsigned int[hash_list_size];
	hash_value_index_minus = new unsigned int[hash_list_size];// index in the genome
	kmer_exist_and_index_minus = new long long int[all_kmer_nums]; // store the hash value index in the sorted hash_value_each_position_point
	unsigned int i;
	long long int num = -1, num2 = -1;													    // and not exist: -1 
	while (getline(libfile, line)) {
		if (line.find("Minus") != string::npos) {
			flag = 0;
			continue;
		}
		istringstream iss(line);
		iss >> encode >> positionsInGenome;
		if (flag == 1) {
			num += 1;
			hash_value_each_position_point[num] = encode;
			hash_value_index[num] = positionsInGenome;
		}
		else {
			num2 += 1;
			hash_value_each_position_point_minus[num2] = encode;
			hash_value_index_minus[num2] = positionsInGenome;
		}
	}
	num++;
	num2++;
	if (num != num2 || num != hash_list_size) {
		bomb_error("The hash encode number in the library file is not equal to the genome length!");
	}

	for (i = 0; i < all_kmer_nums; i++) {
		kmer_exist_and_index[i] = -1;
		kmer_exist_and_index_minus[i] = -1;
	}
	hash_value_is = hash_value_each_position_point[0];
	kmer_exist_and_index[hash_value_is] = 0;
	hash_value_temp = hash_value_is;
	for (i = 1; i < hash_list_size; i++) {
		hash_value_is2 = hash_value_each_position_point[i];
		if (hash_value_is2 == hash_value_temp)
			continue;
		//printf("kmer_exist_and_index[%d] = %d\n", hash_value_is2, i);
		kmer_exist_and_index[hash_value_is2] = i;
		hash_value_temp = hash_value_is2;
		//printf("kmer_exist_and_index[%d] = %lld\n", hash_value_is2, kmer_exist_and_index[hash_value_is2]);
	}
	hash_value_is = hash_value_each_position_point_minus[0];
	kmer_exist_and_index_minus[hash_value_is] = 0;
	hash_value_temp = hash_value_is;
	for (i = 1; i < hash_list_size; i++) {
		hash_value_is2 = hash_value_each_position_point_minus[i];
		if (hash_value_is2 == hash_value_temp)
			continue;
		kmer_exist_and_index_minus[hash_value_is2] = i;
		hash_value_temp = hash_value_is2;
		//printf("kmer_exist_and_index[%d] = %lld\n", hash_value_is2, kmer_exist_and_index[hash_value_is2]);
	}
	return kmer_length;
}



//void GenomeDB::SequenceLocateInGenome_merge_sort(Sequence* seq, const Options & options, unsigned int * hash_value_each_position_point, unsigned int * hash_value_index, long long int * kmer_exist_and_index, unsigned int * hash_value_each_position_point_minus, unsigned int * hash_value_index_minus, long long int * kmer_exist_and_index_minus) {
void GenomeDB::SequenceLocateInGenome_merge_sort(Sequence* seq, const Options & options) {
	NVector<unsigned int> locations, locations2, locat_in_genome, locat_in_genome2, locat_in_sequence, locat_in_sequence2;
	//locat_is2,
	if (seq->index == 220) {
		int aaaaaa = 0;
	}
	int index_modified, interval, location_num, location_num2, nnn, seq_len = seq->size;
	unsigned int temp, encode, k2, j, weight, startt, startt2, index_max, index_max2, num_max, num_max2; // unsigned int:  0～4294967295
	unsigned int start_position_in_sequence, start_position_in_genome;
	long long encode_exist_index_plus, encode_exist_index_minus;
	for (int i = 0; i < seq_len - options.kmer_length + 1; i++) {
		encode = 0;
		k2 = options.kmer_length;
		for (j = 0; j < options.kmer_length; j++) {
			k2--;
			weight = aa2idx_ACGT[seq->data[i + j] - 'A'];
			temp = weight * NAAN_array[k2];
			encode += temp;
		}
		//locat_is = genomeIndexLocatePlus[encode];
		//printf("kmer_exist_and_index size = %ld\n", sizeof(kmer_exist_and_index) / sizeof(kmer_exist_and_index[0]));
		//for (int i = 0; i < 10000000; i++) {			
		//	printf("kmer_exist_and_index[%d] = %lld\n", i, kmer_exist_and_index[i]);
		//}
		//for (int ii = 883101; ii < 883212; ii++) {
		//	printf("kmer_exist_and_index[%d] = %lld\n", ii, kmer_exist_and_index[ii]);
		//}
		//return;
		//printf("%d-th bases, encode is %d, encode exist: %lld, length: %d\n", i, encode, encode_exist_index_plus, seq_len);
		encode_exist_index_plus = kmer_exist_and_index[encode];
		if (encode_exist_index_plus > -1) {
			for (long long v = encode_exist_index_plus; ; v++) {
				if (hash_value_each_position_point[v] != encode)
					break;
				//bomb_error("Hash encode collided in the plus direction!!!");
				index_modified = hash_value_index[v] - i;
				if (index_modified > 0) {// make sure the index_modified > 0
										 //printf("Plus: %d -th bases, encode = %d, kmer_exist_and_index[%lld] = %d\n", i, encode, v, hash_value_each_position_point[v]);
					locations.Append(index_modified);
					locat_in_sequence.Append(i);
					locat_in_genome.Append(hash_value_index[v]);
				}
			}
		}
		// minus direction
		encode_exist_index_minus = kmer_exist_and_index_minus[encode];
		if (encode_exist_index_minus > -1) {
			for (long long v1 = encode_exist_index_minus; ; v1++) {
				if (hash_value_each_position_point_minus[v1] != encode)
					break;
				//bomb_error("Hash encode collided in the minus direction!!!");
				//printf("Minus: %d -th bases, encode = %d, kmer_exist_and_index_minus[%lld] = %d\n", i, encode, v1, hash_value_each_position_point_minus[v1]);
				index_modified = hash_value_index_minus[v1] - i;
				if (index_modified > 0) {// make sure the index_modified > 0
					locations2.Append(index_modified);
					locat_in_sequence2.Append(i);
					locat_in_genome2.Append(hash_value_index_minus[v1]);
				}
			}
		}
	}
	location_num = locations.size;
	location_num2 = locations2.size; // minus direction
									 //printf("\nPlus locations num: %d, minus: %d\n", location_num, location_num2);
	interval = 300;// ceil(seq_len * 0.15);//(seq_len * 0.15);
				   // calculate the number in each interval and find the max one
	num_max = 0; // store the max number for all interval
	num_max2 = 0; // minus direction
	index_max = 0; // store the index of max number
	index_max2 = 0; // minus
	if (location_num > 0 && location_num2 > 0) { // both direction
		unsigned int *locations_number = new unsigned int[location_num]; //.Resize(location_num);
		unsigned int *array_location = new unsigned int[location_num];
		unsigned int *array_location_index = new unsigned int[location_num];
		for (int vv = 0; vv < location_num; vv++) {
			array_location[vv] = locations[vv];
			array_location_index[vv] = vv;
			locations_number[vv] = 0;
		}
		MergeSortRecursionAndIndex_with_matrix_point(array_location, 0, location_num - 1, array_location_index); // ascending order 
		unsigned int *locations_number2 = new unsigned int[location_num2];
		unsigned int *array_location2 = new unsigned int[location_num2];
		unsigned int *array_location_index2 = new unsigned int[location_num2];
		for (int vv = 0; vv < location_num2; vv++) {
			array_location2[vv] = locations2[vv];
			array_location_index2[vv] = vv;
			locations_number2[vv] = 0;
		}
		MergeSortRecursionAndIndex_with_matrix_point(array_location2, 0, location_num2 - 1, array_location_index2); // ascending order 
		for (int ii = 0; ii < location_num; ii++) {
			startt = array_location[ii];
			for (int gg = ii; gg < location_num; gg++) {
				if (array_location[gg] - startt <= interval) {
					locations_number[ii] += 1;
				}
				else {
					break;
				}
			}
			if (num_max < locations_number[ii]) {
				num_max = locations_number[ii];
				index_max = ii;
			}
		}
		for (int ii = 0; ii < location_num2; ii++) {
			startt2 = array_location2[ii];
			for (int gg = ii; gg < location_num2; gg++) {
				if (array_location2[gg] - startt2 <= interval) {
					locations_number2[ii] += 1;
				}
				else {
					break;
				}
			}
			if (num_max2 < locations_number2[ii]) {
				num_max2 = locations_number2[ii];
				index_max2 = ii;
			}
		}
		if (num_max >= num_max2 && num_max2 > 0) { //plus direction
												   // find the start position in genome and sequence
												   //printf("Plus located!\n");
			unsigned int original_index = array_location_index[index_max];
			unsigned int position_in_sequence = locat_in_sequence[original_index];
			unsigned int position_in_genome = locat_in_genome[original_index];
			seq->Set_sequence_locate(1, position_in_sequence, position_in_genome, 0);
		}
		else if (num_max2 > num_max && num_max > 0) {//minus direction
													 //printf("Minus located!\n");
			unsigned int original_index2 = array_location_index2[index_max2];
			unsigned int position_in_sequence2 = locat_in_sequence2[original_index2];
			unsigned int position_in_genome2 = locat_in_genome2[original_index2];
			seq->Set_sequence_locate(0, position_in_sequence2, 0, position_in_genome2);
		}
		else {
			seq->Set_sequence_locate(-1, -1, -1, -1);
		}
		locations.Clear();
		locations2.Clear();
		//locat_is.Clear(); 
		//locat_is2.Clear();
		locat_in_genome.Clear();
		locat_in_genome2.Clear();
		locat_in_sequence.Clear();
		locat_in_sequence2.Clear();
		delete[] locations_number;
		delete[] locations_number2;
		delete[] array_location;
		delete[] array_location2;
		delete[] array_location_index;
		delete[] array_location_index2;
	}
	else if (location_num > 0 && location_num2 == 0) {// no minus direction
		unsigned int *locations_number = new unsigned int[location_num]; //.Resize(location_num);
		unsigned int *array_location = new unsigned int[location_num];
		unsigned int *array_location_index = new unsigned int[location_num];
		for (int vv = 0; vv < location_num; vv++) {
			array_location[vv] = locations[vv];
			array_location_index[vv] = vv;
			locations_number[vv] = 0;
		}
		MergeSortRecursionAndIndex_with_matrix_point(array_location, 0, location_num - 1, array_location_index); // ascending order 
		for (int ii = 0; ii < location_num; ii++) {
			startt = array_location[ii];
			for (int gg = ii; gg < location_num; gg++) {
				if (array_location[gg] - startt <= interval) {
					locations_number[ii] += 1;
				}
				else {
					break;
				}
			}
			if (num_max < locations_number[ii]) {
				num_max = locations_number[ii];
				index_max = ii;
			}
		}
		//printf("Plus located!\n");
		unsigned int original_index = array_location_index[index_max];
		unsigned int position_in_sequence = locat_in_sequence[original_index];
		unsigned int position_in_genome = locat_in_genome[original_index];
		seq->Set_sequence_locate(1, position_in_sequence, position_in_genome, 0);
		locations.Clear();
		locations2.Clear();
		//locat_is.Clear(); 
		//locat_is2.Clear();
		locat_in_genome.Clear();
		locat_in_genome2.Clear();
		locat_in_sequence.Clear();
		locat_in_sequence2.Clear();
		delete[] locations_number;
		delete[] array_location;
		delete[] array_location_index;
	}
	else if (location_num == 0 && location_num2 > 0) { // just minus direction
		unsigned int *locations_number2 = new unsigned int[location_num2];
		unsigned int *array_location2 = new unsigned int[location_num2];
		unsigned int *array_location_index2 = new unsigned int[location_num2];
		for (int vv = 0; vv < location_num2; vv++) {
			array_location2[vv] = locations2[vv];
			array_location_index2[vv] = vv;
			locations_number2[vv] = 0;
		}
		MergeSortRecursionAndIndex_with_matrix_point(array_location2, 0, location_num2 - 1, array_location_index2); // ascending order 
		for (int ii = 0; ii < location_num2; ii++) {
			startt2 = array_location2[ii];
			for (int gg = ii; gg < location_num2; gg++) {
				if (array_location2[gg] - startt2 <= interval) {
					locations_number2[ii] += 1;
				}
				else {
					break;
				}
			}
			if (num_max2 < locations_number2[ii]) {
				num_max2 = locations_number2[ii];
				index_max2 = ii;
			}
		}
		//printf("Minus located!\n");
		unsigned int original_index2 = array_location_index2[index_max2];
		unsigned int position_in_sequence2 = locat_in_sequence2[original_index2];
		unsigned int position_in_genome2 = locat_in_genome2[original_index2];
		seq->Set_sequence_locate(0, position_in_sequence2, 0, position_in_genome2);
		locations.Clear();
		locations2.Clear();
		//locat_is.Clear(); 
		//locat_is2.Clear();
		locat_in_genome.Clear();
		locat_in_genome2.Clear();
		locat_in_sequence.Clear();
		locat_in_sequence2.Clear();
		delete[] locations_number2;
		delete[] array_location2;
		delete[] array_location_index2;
	}
	else {
		seq->Set_sequence_locate(-1, -1, -1, -1);
	}
}
void GenomeDB::SequenceLocateInGenome(Sequence* seq, const Options & options) {
	NVector<unsigned int> locations, locations2, locat_in_genome, locat_in_genome2, locat_in_sequence, locat_in_sequence2, locations_number, locations_number2;
	//locat_is2,
	if (seq->index == 220) {
		int aaaaaa = 0;
	}
	int index_modified, interval, location_num, location_num2, nnn, seq_len = seq->size;
	unsigned int temp, encode, k2, j, weight, startt, startt2, index_max, index_max2, num_max, num_max2; // unsigned int:  0～4294967295
	unsigned int start_position_in_sequence, start_position_in_genome;
	for (int i = 0; i < seq_len - options.kmer_length + 1; i++) {
		encode = 0;
		k2 = options.kmer_length;
		for (j = 0; j < options.kmer_length; j++) {
			k2--;
			weight = aa2idx_ACGT[seq->data[i + j] - 'A'];
			temp = weight * NAAN_array[k2];
			encode += temp;
		}
		//locat_is = genomeIndexLocatePlus[encode];
		for (int v = 0; v < genomeIndexLocatePlus[encode].Size(); v++) {
			index_modified = genomeIndexLocatePlus[encode][v] - i;
			if (index_modified > 0) {// make sure the index_modified > 0
				locations.Append(index_modified);
				locat_in_sequence.Append(i);
				locat_in_genome.Append(genomeIndexLocatePlus[encode][v]);
			}

		}
		// minus direction
		//locat_is2 = genomeIndexLocateMinus[encode];
		for (int v = 0; v < genomeIndexLocateMinus[encode].Size(); v++) {
			index_modified = genomeIndexLocateMinus[encode][v] - i;
			if (index_modified > 0) {// make sure the index_modified > 0
				locations2.Append(index_modified);
				locat_in_sequence2.Append(i);
				locat_in_genome2.Append(genomeIndexLocateMinus[encode][v]);
			}
		}
	}
	location_num = locations.size;
	location_num2 = locations2.size; // minus direction
	locations_number.Resize(location_num);
	locations_number2.Resize(location_num2);
	struct node *array_location = new node[location_num];
	for (int vv = 0; vv < location_num; vv++) {
		array_location[vv].data = locations[vv];
		array_location[vv].index_in_genome = locat_in_genome[vv];
		array_location[vv].index_in_sequence = locat_in_sequence[vv];
		locations_number[vv] = 0;
	}
	qsort(array_location, location_num, sizeof(struct node), comp); // ascending order 

	struct node *array_location2 = new node[location_num2];
	for (int vv = 0; vv < location_num2; vv++) {
		array_location2[vv].data = locations2[vv];
		array_location2[vv].index_in_genome = locat_in_genome2[vv];
		array_location2[vv].index_in_sequence = locat_in_sequence2[vv];
		locations_number2[vv] = 0;
	}
	qsort(array_location2, location_num2, sizeof(struct node), comp); // ascending order 


																	  // write the sorted array to a txt to debug
#if 0
	ofstream outfile;
	outfile.open("D:\\OTU拼接\\回国后文章\\c++代码\\out.txt", ios::out);
	if (!outfile.is_open())
		bomb_error("Open file failure");
	for (int aaa = 0; aaa < location_num; aaa++) {
		outfile << array_location[aaa].data << "\t" << array_location[aaa].index_in_sequence << "\t" << array_location[aaa].index_in_genome << endl;  //在result.txt中写入结果
	}
	outfile.close();
#endif

	interval = 300;// ceil(seq_len * 0.15);//(seq_len * 0.15);
				   // calculate the number in each interval and find the max one
	num_max = 0; // store the max number for all interval
	num_max2 = 0; // minus direction
	index_max = 0; // store the index of max number
	index_max2 = 0; // minus
	for (int ii = 0; ii < location_num; ii++) {
		startt = array_location[ii].data;
		for (int gg = ii; gg < location_num; gg++) {
			if (array_location[gg].data - startt <= interval) {
				locations_number[ii] += 1;
			}
			else {
				break;
			}
		}
		if (num_max < locations_number[ii]) {
			num_max = locations_number[ii];
			index_max = ii;
		}
	}

	for (int ii = 0; ii < location_num2; ii++) {
		startt2 = array_location2[ii].data;
		for (int gg = ii; gg < location_num2; gg++) {
			if (array_location2[gg].data - startt2 <= interval) {
				locations_number2[ii] += 1;
			}
			else {
				break;
			}
		}
		if (num_max2 < locations_number2[ii]) {
			num_max2 = locations_number2[ii];
			index_max2 = ii;
		}
	}
	if (num_max >= num_max2 && num_max2 > 0) { //plus direction
											   // find the start position in genome and sequence
											   // maybe there are more than one position in the sequence, we find the minimum position
											   //seq->plus = 1;
		unsigned int position_min_in_sequence = array_location[index_max].index_in_sequence;
		unsigned int best_index = index_max;
		for (int igg = index_max + 1; ; igg++) {
			if (array_location[igg].data == array_location[index_max].data) {
				if (array_location[igg].index_in_sequence < position_min_in_sequence) {
					position_min_in_sequence = array_location[igg].index_in_sequence;
					best_index = igg;
				}
			}
			else
				break;
		}
		seq->Set_sequence_locate(1, array_location[best_index].index_in_sequence, array_location[best_index].index_in_genome, 0);
		//seq->position_in_sequence = array_location[best_index].index_in_sequence;
		//seq->position_in_genome = array_location[best_index].index_in_genome;
		string time_now = getTime();
		ofstream outfile;
		const char* filename = "positionDensity.txt";
		outfile.open(filename, ios::out);
		//outfile2.open(options.output, ios::app);
		if (!outfile.is_open())
			bomb_error("Open genome library output file failure, exit!");
		//for (int aaa = 0; aaa < location_num; aaa++) {
		outfile << seq->identifier << endl;
		for (int jj = 0; jj < location_num; jj++) {
			outfile << array_location[jj].data << " ";
		}
		outfile << endl;
		for (int jj = 0; jj < location_num; jj++) {
			outfile << array_location[jj].index_in_sequence << " ";
		}
		outfile << endl;
		for (int jj = 0; jj < location_num; jj++) {
			outfile << array_location[jj].index_in_genome << " ";
		}
		outfile << endl;
		for (int jj = 0; jj < location_num; jj++) {
			outfile << locations_number[jj] << " ";
		}
		outfile.close();
		//outfile << "Word length: " << options.kmer_length << endl;
		//outfile << "Direction:   Plus" << endl;
	}
	else if (num_max2 > num_max && num_max > 0) {//minus direction
												 // find the start position in genome and sequence
												 // maybe there are more than one position in the sequence, we find the minimum position
												 //seq->plus = 0;
		unsigned int position_min_in_sequence = array_location2[index_max2].index_in_sequence;
		unsigned int best_index = index_max2;
		for (int igg = index_max2 + 1; ; igg++) {
			if (array_location2[igg].data == array_location2[index_max2].data) {
				if (array_location2[igg].index_in_sequence < position_min_in_sequence) {
					position_min_in_sequence = array_location2[igg].index_in_sequence;
					best_index = igg;
				}
			}
			else
				break;
		}
		seq->Set_sequence_locate(0, array_location2[best_index].index_in_sequence, 0, array_location2[best_index].index_in_genome);
		//seq->position_in_sequence = array_location2[best_index].index_in_sequence;
		//seq->position_in_genome_minus = array_location2[best_index].index_in_genome;
	}
	else {
		seq->Set_sequence_locate(-1, -1, -1, -1);
	}

	delete array_location;
	delete array_location2;
	locations.Clear();
	locations2.Clear();
	//locat_is.Clear(); 
	//locat_is2.Clear();
	locat_in_genome.Clear();
	locat_in_genome2.Clear();
	locat_in_sequence.Clear();
	locat_in_sequence2.Clear();
	locations_number.Clear();
	locations_number2.Clear();
	//printf("Locations number: %d\n", locations.size);
}

void GenomeDB::SequenceLocateInGenome_multi_threads(SequenceDB & seq_db, const Options & options) {
	int tid = 0;
	float p = 0.0, p0 = 0.0;
#pragma omp parallel for schedule( dynamic, 1 )
	for (int i = 0; i < seq_db.sequences.size(); i++) {
		if (seq_db.sequences[i]->size <= 10000) {
			//int aaaa = 0;
			//continue;
		}
		//printf("%d -th in sequence", i);
		SequenceLocateInGenome_merge_sort(seq_db.sequences[i], options); // find the homologous region
																		 //if (seq_db.sequences[i]->plus == 1) {
																		 //	printf("%d-th, position in sequence: %d, in genome: %d\n", i, seq_db.sequences[i]->position_in_sequence, seq_db.sequences[i]->position_in_genome);																//printf("The %d-th sequence position is located\n", i);
																		 //}
																		 //else if(seq_db.sequences[i]->plus == 0) {
																		 //	printf("%d-th, position in sequence: %d, in genome: %d\n", i, seq_db.sequences[i]->position_in_sequence, seq_db.sequences[i]->position_in_genome_minus);																//printf("The %d-th sequence position is located\n", i);
																		 //}
																		 //else {
																		 //	printf("%d-th sequence not located!", i);
																		 //}
		tid = omp_get_thread_num();
		if (omp_get_thread_num() == 0) {
			p = (100.0*i) / seq_db.sequences.size();
			if (p > p0 + 1E-1) {
				printf("\r%5.1f%%   %8d-th sequence", p, i); //printf("The %d-th sequence is located", i);
				p0 = p;
			}
		}
		fflush(stdout);
		//break;
	}
}

void GenomeDB::SequenceLocateInGenome_vector(Sequence* seq, const Options & options) {
	NVector<unsigned int> locations, locations2, locat_in_genome, locat_in_genome2, locat_in_sequence, locat_in_sequence2, locations_number, locations_number2;
	NVector<unsigned int> locat_is, locat_is2;
	int index_modified, interval, location_num, location_num2, nnn, seq_len = seq->size;
	unsigned int temp, encode, k2, j, weight, startt, startt2, index_max, index_max2, num_max, num_max2; // unsigned int:  0～4294967295
	unsigned int start_position_in_sequence, start_position_in_genome;
	for (int i = 0; i < seq_len - options.kmer_length + 1; i++) {
		encode = 0;
		k2 = options.kmer_length;
		for (j = 0; j < options.kmer_length; j++) {
			k2--;
			weight = aa2idx_ACGT[seq->data[i + j] - 'A'];
			temp = weight * NAAN_array[k2];
			encode += temp;
		}
		locat_is = genomeIndexLocatePlus[encode];
		for (int v = 0; v < locat_is.Size(); v++) {
			index_modified = locat_is[v] - i;
			if (index_modified > 0) {// make sure the index_modified > 0
				locations.Append(index_modified);
				locat_in_sequence.Append(i);
				locat_in_genome.Append(locat_is[v]);
			}

		}
		// minus direction
		locat_is2 = genomeIndexLocateMinus[encode];
		for (int v = 0; v < locat_is2.Size(); v++) {
			index_modified = locat_is2[v] - i;
			if (index_modified > 0) {// make sure the index_modified > 0
				locations2.Append(index_modified);
				locat_in_sequence2.Append(i);
				locat_in_genome2.Append(locat_is2[v]);
			}

		}

	}

	location_num = locations.Size();
	location_num2 = locations2.Size(); // minus direction
	locations_number.Resize(location_num);
	locations_number2.Resize(location_num2);
	struct node *array_location = new node[location_num];
	for (int vv = 0; vv < location_num; vv++) {
		array_location[vv].data = locations[vv];
		array_location[vv].index_in_genome = locat_in_genome[vv];
		array_location[vv].index_in_sequence = locat_in_sequence[vv];
		locations_number[vv] = 0;
	}
	qsort(array_location, location_num, sizeof(struct node), comp); // ascending order 

	struct node *array_location2 = new node[location_num2];
	for (int vv = 0; vv < location_num2; vv++) {
		array_location2[vv].data = locations2[vv];
		array_location2[vv].index_in_genome = locat_in_genome2[vv];
		array_location2[vv].index_in_sequence = locat_in_sequence2[vv];
		locations_number2[vv] = 0;
	}
	qsort(array_location2, location_num2, sizeof(struct node), comp); // ascending order 


																	  // write the sorted array to a txt to debug
#if 0
	ofstream outfile;
	outfile.open("D:\\OTU拼接\\回国后文章\\c++代码\\out.txt", ios::out);
	if (!outfile.is_open())
		bomb_error("Open file failure");
	for (int aaa = 0; aaa < location_num; aaa++) {
		outfile << array_location[aaa].data << "\t" << array_location[aaa].index_in_sequence << "\t" << array_location[aaa].index_in_genome << endl;  //在result.txt中写入结果
	}
	outfile.close();
#endif

	interval = 300;// ceil(seq_len * 0.15);//(seq_len * 0.15);
				   // calculate the number in each interval and find the max one
	num_max = 0; // store the max number for all interval
	num_max2 = 0; // minus direction
	index_max = 0; // store the index of max number
	index_max2 = 0; // minus
	for (int ii = 0; ii < location_num; ii++) {
		startt = array_location[ii].data;
		for (int gg = ii; gg < location_num; gg++) {
			if (array_location[gg].data - startt <= interval) {
				locations_number[ii] += 1;
			}
			else {
				break;
			}
		}
		if (num_max < locations_number[ii]) {
			num_max = locations_number[ii];
			index_max = ii;
		}
	}

	for (int ii = 0; ii < location_num2; ii++) {
		startt2 = array_location2[ii].data;
		for (int gg = ii; gg < location_num2; gg++) {
			if (array_location2[gg].data - startt2 <= interval) {
				locations_number2[ii] += 1;
			}
			else {
				break;
			}
		}
		if (num_max2 < locations_number2[ii]) {
			num_max2 = locations_number2[ii];
			index_max2 = ii;
		}
	}
	if (num_max >= num_max2) { //plus direction
							   // find the start position in genome and sequence
							   // maybe there are more than one position in the sequence, we find the minimum position
		seq->plus = 1;
		unsigned int position_min_in_sequence = array_location[index_max].index_in_sequence;
		unsigned int best_index = index_max;
		for (int igg = index_max + 1; ; igg++) {
			if (array_location[igg].data == array_location[index_max].data) {
				if (array_location[igg].index_in_sequence < position_min_in_sequence) {
					position_min_in_sequence = array_location[igg].index_in_sequence;
					best_index = igg;
				}
			}
			else
				break;
		}
		seq->position_in_sequence = array_location[best_index].index_in_sequence;
		seq->position_in_genome = array_location[best_index].index_in_genome;
	}
	else {//minus direction
		  // find the start position in genome and sequence
		  // maybe there are more than one position in the sequence, we find the minimum position
		seq->plus = 0;
		unsigned int position_min_in_sequence = array_location2[index_max2].index_in_sequence;
		unsigned int best_index = index_max2;
		for (int igg = index_max2 + 1; ; igg++) {
			if (array_location2[igg].data == array_location2[index_max2].data) {
				if (array_location2[igg].index_in_sequence < position_min_in_sequence) {
					position_min_in_sequence = array_location2[igg].index_in_sequence;
					best_index = igg;
				}
			}
			else
				break;
		}
		seq->position_in_sequence = array_location2[best_index].index_in_sequence;
		seq->position_in_genome_minus = array_location2[best_index].index_in_genome;
	}
	delete array_location;
	delete array_location2;
	locations.Clear();
	locations2.Clear();
	locat_is.Clear();
	locat_is2.Clear();
	locat_in_genome.Clear();
	locat_in_genome2.Clear();
	locat_in_sequence.Clear();
	locat_in_sequence2.Clear();
	locations_number.Clear();
	locations_number2.Clear();
	//printf("Locations number: %d\n", locations.size);
}

float GenomeDB::seq_align_global_my_no_banded(Sequence *seq1, const Options & options) {
	int len1 = seq1->size - seq1->position_in_sequence + 1;
	int len2 = len1 + 200;
	int gapScore = options.gap_score;
	int matchScore = options.match_score;
	int mismatchScore = options.mismatch_score;
	int **matrix;
	int **track;
	string seq1_aligned = "";
	string seq2_aligned = "";
	string middle = "";
	int from_up, from_diag, from_left, index_in_sequence, index_in_genome;
	int match_num = 0;
	Sequence *genome_is = genome;
	//  up 1, diag 2, left 3
	matrix = new int*[len1]; //设置行 或直接double **data=new double*[m]; 一个指针指向一个指针数组。  
	track = new int*[len1];
	for (int j = 0; j < len1; j++) {
		matrix[j] = new int[len2];        //这个指针数组的每个指针元素又指向一个数组
		track[j] = new int[len2];
	}
	//初始化第一列
	for (int i = 0; i < len1; i++) {
		matrix[i][0] = gapScore * i;
		track[i][0] = 0;
	}
	//初始化第一行
	for (int j = 0; j < len2; j++) {
		matrix[0][j] = gapScore * j;
		track[0][j] = 0;
	}
	for (int i = 1; i < len1; i++) {
		index_in_sequence = seq1->position_in_sequence + i;
		for (int j = 1; j < len2; j++) {
			index_in_genome = seq1->position_in_genome + j;
			if (seq1->data[index_in_sequence - 1] == genome_is->data[index_in_genome - 1])
				from_diag = matrix[i - 1][j - 1] + matchScore;
			else
				from_diag = matrix[i - 1][j - 1] + mismatchScore;
			from_up = matrix[i - 1][j] + gapScore;
			from_left = matrix[i][j - 1] + gapScore;
			if (from_diag >= from_up && from_diag >= from_left) {
				matrix[i][j] = from_diag;
				track[i][j] = 2;//  up 1, diag 2, left 3
			}

			else if (from_up >= from_diag && from_up >= from_left) {
				matrix[i][j] = from_up;
				track[i][j] = 1;//  up 1, diag 2, left 3
			}
			else {
				matrix[i][j] = from_left;
				track[i][j] = 3;//  up 1, diag 2, left 3
			}

		}
	}

	//string haha = "AXNXCXXGXXXXXXXXT"; // A 0, C 4, G 7, T 16, N 2
	int i = len1 - 1;
	int j = len2 - 1;
	for (; i > 0 && j > 0;) {
		index_in_sequence = seq1->position_in_sequence + i;
		index_in_genome = seq1->position_in_genome + j;
		if (track[i][j] == 2) {//  up 1, diag 2, left 3
			seq1_aligned += seq1->data[index_in_sequence];
			seq2_aligned += genome_is->data[index_in_genome];
			if (seq1->data[index_in_sequence] == genome_is->data[index_in_genome]) {
				middle += "|";
				match_num++;
			}
			else
				middle += " ";
			i--;
			j--;
		}
		else if (track[i][j] == 1) {//  up 1, diag 2, left 3
			seq1_aligned += seq1->data[index_in_sequence];
			i--;
			seq2_aligned += "-";//haha.at((int)seq2->data[j]);
			middle += " ";
		}
		else {
			seq1_aligned += "-";//haha.at((int)seq1->data[i - 1]);
			seq2_aligned += genome_is->data[index_in_genome];
			j--;
			middle += " ";
		}
	}
	if (seq1_aligned.size() != seq2_aligned.size()) {
		bomb_error("Sequence aligned error: two aligned sequences lengths are not equal!");
	}
	reverse(seq1_aligned.begin(), seq1_aligned.end());
	reverse(seq2_aligned.begin(), seq2_aligned.end());
	reverse(middle.begin(), middle.end());
#if 0
	cout << seq1_aligned << endl;
	cout << middle << endl;
	cout << seq2_aligned << endl;
#endif


	// release memory
	for (int i = 0; i < len1; i++) {
		delete[] matrix[i]; //先撤销指针元素所指向的数组
		delete[] track[i];
	}
	delete[] matrix;
	delete[] track;

	return match_num / (float)seq1_aligned.size();
}

float GenomeDB::seq_align_local_my_no_banded(Sequence *seq1, const Options & options) {
	int len1 = seq1->size - seq1->position_in_sequence + 1;
	int len2 = len1 + 200;
	int gapScore = options.gap_score;
	int matchScore = options.match_score;
	int mismatchScore = options.mismatch_score;
	int **matrix;
	int **track;
	int max_score = 0, max_i = 0, max_j = 0; // store the max score and row, column nymber
	string seq1_aligned = "";
	string seq2_aligned = "";
	string middle = "";
	int from_up, from_diag, from_left, index_in_sequence, index_in_genome;
	int match_num = 0;
	Sequence *genome_is = genome;
	//  up 1, diag 2, left 3
	matrix = new int*[len1]; //设置行 或直接double **data=new double*[m]; 一个指针指向一个指针数组。  
	track = new int*[len1];
	for (int j = 0; j < len1; j++) {
		matrix[j] = new int[len2];        //这个指针数组的每个指针元素又指向一个数组
		track[j] = new int[len2];
	}
	//初始化第一列
	for (int i = 0; i < len1; i++) {
		matrix[i][0] = 0;
		track[i][0] = 0;
	}
	//初始化第一行
	for (int j = 0; j < len2; j++) {
		matrix[0][j] = 0;
		track[0][j] = 0;
	}
	for (int i = 1; i < len1; i++) {
		index_in_sequence = seq1->position_in_sequence + i;
		for (int j = 1; j < len2; j++) {
			index_in_genome = seq1->position_in_genome + j;
			if (seq1->data[index_in_sequence - 1] == genome_is->data[index_in_genome - 1])
				from_diag = matrix[i - 1][j - 1] + matchScore;
			else
				from_diag = matrix[i - 1][j - 1] + mismatchScore;
			from_up = matrix[i - 1][j] + gapScore;
			from_left = matrix[i][j - 1] + gapScore;
			if (from_diag >= from_up && from_diag >= from_left && from_diag >= 0) {
				matrix[i][j] = from_diag;
				track[i][j] = 2;//  up 1, diag 2, left 3
				if (from_diag > max_score) {
					max_score = from_diag;
					max_i = i;
					max_j = j;
				}
			}

			else if (from_up >= from_diag && from_up >= from_left && from_up >= 0) {
				matrix[i][j] = from_up;
				track[i][j] = 1;//  up 1, diag 2, left 3
				if (from_up > max_score) {
					max_score = from_up;
					max_i = i;
					max_j = j;
				}
			}
			else if (from_left >= from_diag && from_left >= from_up && from_left >= 0) {
				matrix[i][j] = from_left;
				track[i][j] = 3;//  up 1, diag 2, left 3
				if (from_left > max_score) {
					max_score = from_left;
					max_i = i;
					max_j = j;
				}
			}
			else {
				matrix[i][j] = 0;
				track[i][j] = 0;
			}

		}
	}

	int i = max_i - 1;// len1 - 1;
	int j = max_j - 1;// len2 - 1;
	for (; i > 0 && j > 0;) {
		index_in_sequence = seq1->position_in_sequence + i;
		index_in_genome = seq1->position_in_genome + j;
		if (track[i][j] == 2) {//  up 1, diag 2, left 3
			seq1_aligned += seq1->data[index_in_sequence];
			seq2_aligned += genome_is->data[index_in_genome];
			if (seq1->data[index_in_sequence] == genome_is->data[index_in_genome]) {
				middle += "|";
				match_num++;
			}
			else
				middle += " ";
			i--;
			j--;
		}
		else if (track[i][j] == 1) {//  up 1, diag 2, left 3
			seq1_aligned += seq1->data[index_in_sequence];
			i--;
			seq2_aligned += "-";//haha.at((int)seq2->data[j]);
			middle += " ";
		}
		else {
			seq1_aligned += "-";//haha.at((int)seq1->data[i - 1]);
			seq2_aligned += genome_is->data[index_in_genome];
			j--;
			middle += " ";
		}
	}
	if (seq1_aligned.size() != seq2_aligned.size()) {
		bomb_error("Sequence aligned error: two aligned sequences lengths are not equal!");
	}
	reverse(seq1_aligned.begin(), seq1_aligned.end());
	reverse(seq2_aligned.begin(), seq2_aligned.end());
	reverse(middle.begin(), middle.end());

#if 0
	ofstream outfile;
	outfile.open("D:\\OTU拼接\\回国后文章\\c++代码\\out.txt", ios::out);
	if (!outfile.is_open())
		bomb_error("Open file failure");
	//for (int aaa = 0; aaa < location_num; aaa++) {
	outfile << seq1_aligned << endl;
	outfile << middle << endl;
	outfile << seq2_aligned << endl;
	//outfile << array_location[aaa].index_in_sequence << "\t" << array_location[aaa].index_in_genome << endl;  //在result.txt中写入结果
	//}
	outfile.close();
#endif

	// release memory
	for (int i = 0; i < len1; i++) {
		delete[] matrix[i]; //先撤销指针元素所指向的数组
		delete[] track[i];
	}
	delete[] matrix;
	delete[] track;

	return match_num / (float)seq1_aligned.size();
}

float GenomeDB::seq_align_local_my_with_banded(Sequence *seq1, const Options & options) {
	int len1 = seq1->size;// -seq1->position_in_sequence + 1;
	int len2 = ceil(len1 * 1.15);// 667;
	double coefficient = 1.15;// 0 * len2 / len1;
	int gapScore = options.gap_score;
	int matchScore = options.match_score;
	int mismatchScore = options.mismatch_score;
	int **matrix;
	int **track;
	int max_score = 0, max_i = 0, max_j = 0; // store the max score and row, column number
	int max_score2 = 0, max_i2 = 0, max_j2 = 0; // store the max score and row, column number at the len1-th matrixScore
	int max_score_final = 0, max_i_final = 0, max_j_final = 0;
	string seq1_aligned = "";
	string seq2_aligned = "";
	string middle = "";
	int from_up, from_diag, from_left, index_in_sequence, index_in_genome, i, j;
	int match_num = 0, delete_num = 0, insert_num = 0;
	int banded_left, banded_right;// , match_number = 0;
	Sequence *genome_is;
	if (seq1->plus) {
		genome_is = genome;
	}
	else {
		genome_is = genome_minus;
	}

	//  up 1, diag 2, left 3
	matrix = new int*[len1 + 1]; //设置行 或直接double **data=new double*[m]; 一个指针指向一个指针数组。  
	track = new int*[len1 + 1];
	for (int j = 0; j <= len1; j++) {
		printf("j = %d\n", j);
		matrix[j] = new int[len2 + 1]();    // initialize each value to 0 这个指针数组的每个指针元素又指向一个数组
		track[j] = new int[len2 + 1]();
	}
#if 0
	//初始化第一列
	for (int i = 0; i < len1; i++) {
		matrix[i][0] = 0;
		track[i][0] = 0;
	}
	//初始化第一行
	for (int j = 0; j < len2; j++) {
		matrix[0][j] = 0;
		track[0][j] = 0;
	}
#endif
	int index_in_genome_start;
	int banded_width = ceil(len1 * 0.1);
	if (seq1->plus) {
		index_in_genome_start = seq1->position_in_genome - ceil(seq1->position_in_sequence * 1.15);
	}
	else {
		index_in_genome_start = seq1->position_in_genome_minus - ceil(seq1->position_in_sequence * 1.15);
	}
	for (i = 1; i <= len1; i++) {
		index_in_sequence = i - 1;///seq1->position_in_sequence + i;
		banded_left = max(1, ceil(coefficient * i - banded_width));
		banded_right = min(len2, ceil(coefficient * i + banded_width));
		for (j = banded_left; j <= banded_right; j++) {
			index_in_genome = index_in_genome_start + j - 1;
			//cout << "seq[" << index_in_sequence - 1 << "] = " << seq1->data[index_in_sequence - 1] << endl;
			//cout << "gen[" << index_in_genome - 1 << "} = " << genome_is->data[index_in_genome - 1] << "---------" <<endl;
			if (seq1->data[index_in_sequence] == genome_is->data[index_in_genome])
				from_diag = matrix[i - 1][j - 1] + matchScore;
			else
				from_diag = matrix[i - 1][j - 1] + mismatchScore;
			from_up = matrix[i - 1][j] + gapScore;
			from_left = matrix[i][j - 1] + gapScore;
			if (from_diag >= from_up && from_diag >= from_left && from_diag > 0) {
				matrix[i][j] = from_diag;
				track[i][j] = 2;//  up 1, diag 2, left 3
				if (from_diag > max_score) {
					max_score = from_diag;
					max_i = i;
					max_j = j;
				}
			}

			else if (from_up >= from_diag && from_up >= from_left && from_up > 0) {
				matrix[i][j] = from_up;
				track[i][j] = 1;//  up 1, diag 2, left 3
				if (from_up > max_score) {
					max_score = from_up;
					max_i = i;
					max_j = j;
				}
			}
			else if (from_left >= from_diag && from_left >= from_up && from_left > 0) {
				matrix[i][j] = from_left;
				track[i][j] = 3;//  up 1, diag 2, left 3
				if (from_left > max_score) {
					max_score = from_left;
					max_i = i;
					max_j = j;
				}
			}
			else {
				matrix[i][j] = 0;
				track[i][j] = 0;
			}
			//cout << "M[" << i << "][" << j << "] = " << matrix[i][j] << endl;

		}
		//printf("max_score = %d, i = %d, j = %d\n", max_score, max_i, max_j);
	}
	if (max_i < len1) {
		for (int jjjj = 1; jjjj <= len2; jjjj++) {
			if (matrix[len1][jjjj] > max_score2) {
				max_score2 = matrix[len1][jjjj];
				max_j2 = jjjj;
			}
		}
		if (max_score2 > 0) {
			max_score_final = max_score2;
			max_i_final = len1;
			max_j_final = max_j2;
		}
		else {
			max_score_final = max_score;
			max_i_final = max_i;
			max_j_final = max_j;
		}
	}
	else {
		max_i_final = max_i;
		max_j_final = max_j;
	}

	i = max_i_final;// len1 - 1;
	j = max_j_final;// len2 - 1;
	for (; i > 0 && j > 0;) {
		index_in_sequence = i - 1;
		index_in_genome = index_in_genome_start + j - 1;
		if (track[i][j] == 2) {//  up 1, diag 2, left 3
			seq1_aligned = seq1->data[index_in_sequence] + seq1_aligned;
			seq2_aligned = genome_is->data[index_in_genome] + seq2_aligned;
			if (seq1->data[index_in_sequence] == genome_is->data[index_in_genome]) {
				middle = "|" + middle;
				match_num++;
			}
			else
				middle = " " + middle;
			i--;
			j--;
		}
		else if (track[i][j] == 1) {//  up 1, diag 2, left 3
			seq1_aligned = seq1->data[index_in_sequence] + seq1_aligned;
			i--;
			seq2_aligned = "-" + seq2_aligned;//haha.at((int)seq2->data[j]);
			middle = " " + middle;
			insert_num++;
		}
		else {
			seq1_aligned = "-" + seq1_aligned;//haha.at((int)seq1->data[i - 1]);
			seq2_aligned = genome_is->data[index_in_genome] + seq2_aligned;
			j--;
			middle = " " + middle;
			delete_num++;
		}
	}
	if (seq1_aligned.size() != seq2_aligned.size()) {
		bomb_error("Sequence aligned error: two aligned sequences lengths are not equal!");
	}
	//reverse(seq1_aligned.begin(), seq1_aligned.end());
	//reverse(seq2_aligned.begin(), seq2_aligned.end());
	//reverse(middle.begin(), middle.end());

#if 0
	ofstream outfile;
	outfile.open("D:\\OTU拼接\\回国后文章\\c++代码\\out.txt", ios::out);
	if (!outfile.is_open())
		bomb_error("Open file failure");
	//for (int aaa = 0; aaa < location_num; aaa++) {
	outfile << seq1_aligned << endl;
	outfile << middle << endl;
	outfile << seq2_aligned << endl;
	//outfile << array_location[aaa].index_in_sequence << "\t" << array_location[aaa].index_in_genome << endl;  //在result.txt中写入结果
	//}
	outfile.close();
#endif

	// above alignment is one direction, here is another direction.
	// 
	//int len_seq2 = 

	// release memory
	for (int i = 0; i <= len1; i++) {
		delete[] matrix[i]; //先撤销指针元素所指向的数组
		delete[] track[i];
	}
	delete[] matrix;
	delete[] track;

	return match_num / (float)seq1_aligned.size();
}

int GenomeDB::seq_align_multi_threads(SequenceDB & seqs_db, const Options & options) {
	printf("\n");
	int seq_num = seqs_db.sequences.size();
	int tid = 0;
	float p = 0.0, p0 = 0.0;
	if (options.model.compare("banded") == 0) {
#pragma omp parallel for schedule( dynamic, 1 )
		for (int i = 0; i < seq_num; i++) {
			//if (i != 13) {
			//	continue;
			//}
			float simm = one_seq_align_local_diagonal_banded_for_multi_threads(seqs_db.sequences[i], options);
			//printf("i = %d, identify = %f\n", i, simm);
			tid = omp_get_thread_num();
			if (tid == 0) {
				p = (100.0*i) / seq_num;
				if (p > p0 + 1E-1) {
					printf("\r%5.1f%%,   %8d-th sequence, identify = %5.3f", p, i, simm); //printf("The %d-th sequence is located", i);
					p0 = p;
				}
			}
			fflush(stdout);
		}
	}
	else if (options.model.compare("edlib") == 0) {
#pragma omp parallel for schedule( dynamic, 1 )
		for (int i = 0; i < seq_num; i++) {
//			if (i == 464) {
//				int jj = 0;
//				//continue;
//			}
			float simm = one_seq_align_edlib_for_multi_threads(seqs_db.sequences[i], options);
			//printf("i = %d, identify = %f\n", i, simm);
			tid = omp_get_thread_num();
			if (0 == 0) {
				p = (100.0*i) / seq_num;
				if (p > p0 + 1E-1) {
					printf("\r%5.1f%%,   %8d-th sequence, identify = %5.3f", p, i, simm); //printf("The %d-th sequence is located", i);
					p0 = p;
				}
			}
			fflush(stdout);
			
		}

	}
	return 0;
}

float GenomeDB::seq_align_local_diagonal_banded(Sequence *seq1, const Options & options) {
	// to avoid storing big whole score matrix
	////////////////////////////// first direction:
	/////////////////////////////  ACCGGTAAAAATCGGCCGAAGCAGAT------>
	/////////////////////////////  ||||||||||||| | || |||||||
	/////////////////////////////  ACCGGTAAAAATCTGGCGGAGCAGAT------>
	int len1 = seq1->size - seq1->position_in_sequence;
	char* seq_sub = new char[len1 + 1]; //"TTGATTGATGCAGT";// new char[len1];
	strcpy(seq_sub, seq1->data + seq1->position_in_sequence);
	seq_sub[len1] = '\0';
	double coefficient = 1.08;// 0 * len2 / len1;
	int len2 = ceil(len1 * coefficient) + 5;// 667;	
	int gapScore = options.gap_score;
	int matchScore = options.match_score;
	int mismatchScore = options.mismatch_score;
	int **matrix; // score matrix
	int **track; // track score
	int *baseIndex; // base index 
	int banded_number_last; // banded/aligned number
	int max_score = 0, max_i = 0, max_j = 0; // store the max score and row, column number
	int max_score2 = 0, max_i2 = 0, max_j2 = 0; // store the max score and row, column number at the len1-th matrixScore
	int max_score_final = 0, max_i_final = 0, max_j_final = 0;
	string seq1_aligned = "";
	string seq2_aligned = "";
	string middle = "";
	string direction = "";
	int from_up, from_diag, from_left, index_in_sequence, index_in_genome, i, j;
	int match_num = 0, delete_num = 0, insert_num = 0, subsitute_num = 0;
	int banded_up, banded_down, banded_width;// , match_number = 0;
	if (len1 < 8000) {
		banded_width = ceil(len1 / 5.0);
	}
	else {
		banded_width = ceil((len2 - len1) * 1.3);
	}
	//printf("len1 = %d, band width = %d\n", len1, banded_width);
	int seq1_position_in_genome = 0;
	Sequence *genome_is;
	if (seq1->plus) {
		genome_is = genome;
		seq1_position_in_genome = seq1->position_in_genome;
		direction = "Plus";
	}
	else {
		genome_is = genome_minus;
		seq1_position_in_genome = seq1->position_in_genome_minus;
		direction = "Minus";
	}
	char* genome_sub = new char[len2 + 1]; //"TTTGATTGATGGCAGT";//
	for (int i = 0; i < len2; i++) {
		genome_sub[i] = genome_is->data[seq1_position_in_genome + i];
	}
	genome_sub[len2] = '\0';
	//cout << genome_sub << endl;

	//  up 1, diag 2, left 3
	matrix = new int*[len1 + 1]; //设置行 或直接double **data=new double*[m]; 一个指针指向一个指针数组。  
	track = new int*[len1 + 1];
	baseIndex = new int[len1 + 1]();
	for (int j = 0; j <= len1; j++) {
		//printf("j = %d\n", j);
		matrix[j] = new int[2 * banded_width + 5]();    // initialize each value to 0 这个指针数组的每个指针元素又指向一个数组
		track[j] = new int[2 * banded_width + 5]();
	}
	for (i = 1; i <= len1; i++) {
		index_in_sequence = i - 1;///seq1->position_in_sequence + i;
		banded_up = min(len2, ceil(coefficient * i + banded_width));
		banded_down = max(1, floor(coefficient * i - banded_width));
		baseIndex[i] = max(0, banded_down);
		for (j = banded_down; j <= banded_up; j++) {
			index_in_genome = j - 1;
			//cout << "seq[" << index_in_sequence - 1 << "] = " << seq1->data[index_in_sequence - 1] << endl;
			//cout << "gen[" << index_in_genome - 1 << "} = " << genome_is->data[index_in_genome - 1] << "---------" <<endl;
			if (seq_sub[index_in_sequence] == genome_sub[index_in_genome]) {
				if (j - 1 - baseIndex[i - 1] >= 0) {
					from_diag = matrix[i - 1][j - 1 - baseIndex[i - 1]] + matchScore;
				}
				else {
					from_diag = 2;
				}

			}
			else {
				if (j - 1 - baseIndex[i - 1] >= 0) {
					from_diag = matrix[i - 1][j - 1 - baseIndex[i - 1]] + mismatchScore;
				}
				else {
					from_diag = -2;
				}

			}
			from_up = matrix[i - 1][j - baseIndex[i - 1]] + gapScore;
			from_left = max(0, matrix[i][j - 1 - baseIndex[i]] + gapScore);
			if (from_diag >= from_up && from_diag >= from_left && from_diag > 0) {
				matrix[i][j - baseIndex[i]] = from_diag;
				track[i][j - baseIndex[i]] = 2;//  up 1, diag 2, left 3
				if (from_diag > max_score) {
					max_score = from_diag;
					max_i = i;
					max_j = j;
				}
			}

			else if (from_up >= from_diag && from_up >= from_left && from_up > 0) {
				matrix[i][j - baseIndex[i]] = from_up;
				track[i][j - baseIndex[i]] = 1;//  up 1, diag 2, left 3
				if (from_up > max_score) {
					max_score = from_up;
					max_i = i;
					max_j = j;
				}
			}
			else if (from_left >= from_diag && from_left >= from_up && from_left > 0) {
				matrix[i][j - baseIndex[i]] = from_left;
				track[i][j - baseIndex[i]] = 3;//  up 1, diag 2, left 3
				if (from_left > max_score) {
					max_score = from_left;
					max_i = i;
					max_j = j;
				}
			}
			else {
				matrix[i][j - baseIndex[i]] = 0;
				track[i][j - baseIndex[i]] = 0;
			}
			//cout << "M[" << i << "][" << j << "] = " << matrix[i][j] << endl;

		}
		//printf("max_score = %d, i = %d, j = %d\n", max_score, max_i, max_j);
	}
	banded_number_last = banded_up - banded_down;
	if (max_i < len1) {
		for (int jjjj = 0; jjjj < banded_number_last; jjjj++) {
			if (matrix[len1][jjjj] > max_score2) {
				max_score2 = matrix[len1][jjjj];
				max_j2 = jjjj + baseIndex[len1];
			}
		}
		if (max_score2 > 0) {
			max_score_final = max_score2;
			max_i_final = len1;
			max_j_final = max_j2;
		}
		else {
			max_score_final = max_score;
			max_i_final = max_i;
			max_j_final = max_j;
		}
	}
	else {
		max_i_final = max_i;
		max_j_final = max_j;
	}
	i = max_i_final;// len1 - 1;
	j = max_j_final;// len2 - 1;
	for (; i > 0 && j > 0;) {
		//index_in_sequence = i - 1;
		//index_in_genome = j - 1;
		if (track[i][j - baseIndex[i]] == 2) {//  up 1, diag 2, left 3
			seq1_aligned = seq_sub[i - 1] + seq1_aligned;
			seq2_aligned = genome_sub[j - 1] + seq2_aligned;
			if (seq_sub[i - 1] == genome_sub[j - 1]) {
				middle = "|" + middle;
			}
			else
				middle = " " + middle;
			i--;
			j--;
		}
		else if (track[i][j - baseIndex[i]] == 1) {//  up 1, diag 2, left 3
			seq1_aligned = seq_sub[i - 1] + seq1_aligned;
			i--;
			seq2_aligned = "-" + seq2_aligned;//haha.at((int)seq2->data[j]);
			middle = " " + middle;
		}
		else {
			seq1_aligned = "-" + seq1_aligned;//haha.at((int)seq1->data[i - 1]);
			seq2_aligned = genome_sub[j - 1] + seq2_aligned;
			j--;
			middle = " " + middle;
		}
	}
	if (seq1_aligned.size() != seq2_aligned.size()) {
		bomb_error("Sequence aligned error: two aligned sequences lengths are not equal!");
	}
#if 0
	ofstream outfile;
	outfile.open("D:\\OTU拼接\\回国后文章\\c++代码\\out.txt", ios::out);
	if (!outfile.is_open())
		bomb_error("Open file failure");
	//for (int aaa = 0; aaa < location_num; aaa++) {
	outfile << seq1_aligned << endl;
	outfile << middle << endl;
	outfile << seq2_aligned << endl;
	//outfile << array_location[aaa].index_in_sequence << "\t" << array_location[aaa].index_in_genome << endl;  //在result.txt中写入结果
	//}
	outfile.close();
#endif
	// release memory
	for (int i = 0; i <= len1; i++) {
		delete[] matrix[i];
		delete[] track[i];
	}
	delete[] matrix;
	delete[] track;
	delete[] baseIndex;
	delete[] genome_sub;
	delete[] seq_sub;
	int aaaa = 0;
	//--------------------------------------------------------------------------------------
	////////////////////////////// second direction:
	/////////////////////////////  <-------ACCGGTAAAAA  TCGGCCGAAGCAGAT
	/////////////////////////////          |||||||||||  || | || |||||||
	/////////////////////////////  <-------ACCGGTAAAAA  TCTGGCGGAGCAGAT
	int **matrix2; // score matrix
	int **track2; // track score
	int *baseIndex2; // base index 
	max_score = 0, max_i = 0, max_j = 0; // store the max score and row, column number
	max_score2 = 0, max_i2 = 0, max_j2 = 0; // store the max score and row, column number at the len1-th matrixScore
	max_score_final = 0, max_i_final = 0, max_j_final = 0;
	string seq1_aligned2 = "";
	string seq2_aligned2 = "";
	string middle2 = "";
	//int match_num2 = 0, delete_num2 = 0, insert_num2 = 0;
	len1 = seq1->position_in_sequence + 6;
	seq_sub = new char[len1]; //"TTGATTGATGCAGT";// new char[len1];
	for (int i = 1; i <= len1; i++) {
		seq_sub[i - 1] = seq1->data[len1 - i];
	}
	len2 = ceil(len1 * coefficient) + 5;// 667;
	genome_sub = new char[len2 + 1]; //"TTTGATTGATGGCAGT";//
	for (int i = 0; i < len2; i++) {
		genome_sub[i] = genome_is->data[seq1_position_in_genome - i + 5];
	}
	genome_sub[len2] = '\0';
	if (len1 < 8000) {
		banded_width = ceil(len1 / 5.0);
	}
	else {
		banded_width = ceil((len2 - len1) * 1.3);
	}
	//banded_width = max(ceil(len1/2.0), ceil((len2 - len1) * 1.3));
	//cout << genome_sub << endl;
	//printf("len1 = %d, band width = %d\n", len1, banded_width);

	//  up 1, diag 2, left 3
	matrix2 = new int*[len1 + 1]; //设置行 或直接double **data=new double*[m]; 一个指针指向一个指针数组。  
	track2 = new int*[len1 + 1];
	baseIndex2 = new int[len1 + 1]();
	for (int j = 0; j <= len1; j++) {
		//printf("j = %d\n", j);
		matrix2[j] = new int[2 * banded_width + 5]();    // initialize each value to 0 这个指针数组的每个指针元素又指向一个数组
		track2[j] = new int[2 * banded_width + 5]();
	}
	for (i = 1; i <= len1; i++) {
		index_in_sequence = i - 1;///seq1->position_in_sequence + i;
		banded_up = min(len2, ceil(coefficient * i + banded_width));
		banded_down = max(1, floor(coefficient * i - banded_width));
		baseIndex2[i] = max(0, banded_down);
		for (j = banded_down; j <= banded_up; j++) {
			index_in_genome = j - 1;
			//cout << "seq[" << index_in_sequence - 1 << "] = " << seq1->data[index_in_sequence - 1] << endl;
			//cout << "gen[" << index_in_genome - 1 << "} = " << genome_is->data[index_in_genome - 1] << "---------" <<endl;
			if (seq_sub[index_in_sequence] == genome_sub[index_in_genome]) {
				if (j - 1 - baseIndex2[i - 1] >= 0) {
					from_diag = matrix2[i - 1][j - 1 - baseIndex2[i - 1]] + matchScore;
				}
				else {
					from_diag = 2;
				}

			}
			else {
				if (j - 1 - baseIndex2[i - 1] >= 0) {
					from_diag = matrix2[i - 1][j - 1 - baseIndex2[i - 1]] + mismatchScore;
				}
				else {
					from_diag = -2;
				}

			}
			from_up = matrix2[i - 1][j - baseIndex2[i - 1]] + gapScore;
			from_left = max(0, matrix2[i][j - 1 - baseIndex2[i]] + gapScore);
			if (from_diag >= from_up && from_diag >= from_left && from_diag > 0) {
				matrix2[i][j - baseIndex2[i]] = from_diag;
				track2[i][j - baseIndex2[i]] = 2;//  up 1, diag 2, left 3
				if (from_diag > max_score) {
					max_score = from_diag;
					max_i = i;
					max_j = j;
				}
			}

			else if (from_up >= from_diag && from_up >= from_left && from_up > 0) {
				matrix2[i][j - baseIndex2[i]] = from_up;
				track2[i][j - baseIndex2[i]] = 1;//  up 1, diag 2, left 3
				if (from_up > max_score) {
					max_score = from_up;
					max_i = i;
					max_j = j;
				}
			}
			else if (from_left >= from_diag && from_left >= from_up && from_left > 0) {
				matrix2[i][j - baseIndex2[i]] = from_left;
				track2[i][j - baseIndex2[i]] = 3;//  up 1, diag 2, left 3
				if (from_left > max_score) {
					max_score = from_left;
					max_i = i;
					max_j = j;
				}
			}
			else {
				matrix2[i][j - baseIndex2[i]] = 0;
				track2[i][j - baseIndex2[i]] = 0;
			}
			//cout << "M[" << i << "][" << j << "] = " << matrix[i][j] << endl;

		}
		//printf("max_score = %d, i = %d, j = %d\n", max_score, max_i, max_j);
	}
	banded_number_last = banded_up - banded_down;
	if (max_i < len1) {
		for (int jjjj = 0; jjjj < banded_number_last; jjjj++) {
			if (matrix2[len1][jjjj] > max_score2) {
				max_score2 = matrix2[len1][jjjj];
				max_j2 = jjjj + baseIndex2[len1];
			}
		}
		if (max_score2 > 0) {
			max_score_final = max_score2;
			max_i_final = len1;
			max_j_final = max_j2;
		}
		else {
			max_score_final = max_score;
			max_i_final = max_i;
			max_j_final = max_j;
		}
	}
	else {
		max_i_final = max_i;
		max_j_final = max_j;
	}
	i = max_i_final;// len1 - 1;
	j = max_j_final;// len2 - 1;
	for (; i > 0 && j > 0;) {
		//index_in_sequence = i - 1;
		//index_in_genome = j - 1;
		if (track2[i][j - baseIndex2[i]] == 2) {//  up 1, diag 2, left 3
			seq1_aligned2 = seq_sub[i - 1] + seq1_aligned2;
			seq2_aligned2 = genome_sub[j - 1] + seq2_aligned2;
			if (seq_sub[i - 1] == genome_sub[j - 1]) {
				middle2 = "|" + middle2;
			}
			else
				middle2 = " " + middle2;
			i--;
			j--;
		}
		else if (track2[i][j - baseIndex2[i]] == 1) {//  up 1, diag 2, left 3
			seq1_aligned2 = seq_sub[i - 1] + seq1_aligned2;
			i--;
			seq2_aligned2 = "-" + seq2_aligned2;//haha.at((int)seq2->data[j]);
			middle2 = " " + middle2;
		}
		else {
			seq1_aligned2 = "-" + seq1_aligned2;//haha.at((int)seq1->data[i - 1]);
			seq2_aligned2 = genome_sub[j - 1] + seq2_aligned2;
			j--;
			middle2 = " " + middle2;
		}
	}
	if (seq1_aligned2.size() != seq2_aligned2.size()) {
		bomb_error("Sequence aligned error: two aligned sequences lengths are not equal!");
	}
	reverse(seq1_aligned2.begin(), seq1_aligned2.end());
	reverse(seq2_aligned2.begin(), seq2_aligned2.end());
	reverse(middle2.begin(), middle2.end());
	string seq1_whole_aligned = seq1_aligned2.substr(0, seq1_aligned2.size() - 6) + seq1_aligned;
	string seq2_whole_aligned = seq2_aligned2.substr(0, seq2_aligned2.size() - 6) + seq2_aligned;
	string middle_whole_aligned = middle2.substr(0, middle2.size() - 6) + middle;
	match_num = count(middle_whole_aligned.begin(), middle_whole_aligned.end(), '|');
	delete_num = count(seq1_whole_aligned.begin(), seq1_whole_aligned.end(), '-');
	insert_num = count(seq2_whole_aligned.begin(), seq2_whole_aligned.end(), '-');
	subsitute_num = seq1_whole_aligned.size() - delete_num - subsitute_num - match_num;
	ofstream outfile2;
	const char* filename = options.output.data();
	outfile2.open(filename, ios::app);
	if (!outfile2.is_open())
		bomb_error("Open output file failure");
	//for (int aaa = 0; aaa < location_num; aaa++) {
	outfile2 << "Query:      " << seq1->identifier << endl;
	outfile2 << "Length:     " << seq1->size << endl;
	outfile2 << "nMatch:     " << match_num << endl;
	outfile2 << "nSubsitute: " << subsitute_num << endl;
	outfile2 << "nDelete:    " << delete_num << endl;
	outfile2 << "nInsert:    " << insert_num << endl;
	outfile2 << "Identify:   " << match_num / (float)seq1_whole_aligned.size() << endl;
	//outfile2 << "Score:      " << insert_num << endl;
	outfile2 << "Strand:     Plus/" << direction << endl;
	outfile2 << endl;
	int leftPosition = 0, rightPosition = 0;
	for (int iii = 0; iii < seq1_whole_aligned.size(); ) {
		if (leftPosition >= seq1_whole_aligned.size()) {
			break;
		}
		outfile2 << "Query  " << seq1_whole_aligned.substr(leftPosition, 60) << endl;
		outfile2 << "       " << middle_whole_aligned.substr(leftPosition, 60) << endl;
		outfile2 << "Sbjct  " << seq2_whole_aligned.substr(leftPosition, 60) << endl;
		outfile2 << endl;
		leftPosition += 60;
	}
	outfile2 << endl;
	outfile2 << endl;
	//outfile2 << seq1_whole_aligned << endl;
	//outfile2 << middle_whole_aligned << endl;
	//outfile2 << seq2_whole_aligned << endl;
	//outfile << array_location[aaa].index_in_sequence << "\t" << array_location[aaa].index_in_genome << endl;  //在result.txt中写入结果
	//}
	outfile2.close();
	//#endif
	// release memory
	for (int i = 0; i <= len1; i++) {
		delete[] matrix2[i];
		delete[] track2[i];
	}
	delete[] matrix2;
	delete[] track2;
	delete[] baseIndex2;
	delete[] genome_sub;
	delete[] seq_sub;
	return match_num / (float)seq1_whole_aligned.size();
}

float GenomeDB::one_seq_align_edlib_for_multi_threads(Sequence *seq1, const Options & options) {
	if (seq1->plus == -1) {// alignment position not found for too short
		return 0.0;
	}

	//----------------------------------- forward direction

	int len1 = seq1->size - seq1->position_in_sequence;
//	if (len1 <= 0){
//		int hahaha = 0;
//	}
	char* seq_sub = new char[len1 + 1]; //"TTGATTGATGCAGT";// new char[len1];
	strcpy(seq_sub, seq1->data + seq1->position_in_sequence);
	seq_sub[len1] = '\0';
	double coefficient = 1.08;// 0 * len2 / len1;
	int len2 = ceil(len1 * coefficient) + 5;// 667;	
	int seq1_position_in_genome = 0;
	int genome_id = seq1->tar_id;
	//Sequence genome_is;
	//string direction = "";
	unsigned int genome_size;
	if (seq1->plus) {
		//genome_is = genome[genome_id];
		seq1_position_in_genome = seq1->position_in_genome;
		//direction = "Plus";
		genome_size = genome[genome_id].size;
	}
	else {
		//genome_is = genome_minus_sets[genome_id];
		seq1_position_in_genome = seq1->position_in_genome_minus;
		//direction = "Minus";
		genome_size = genome_minus_sets[genome_id].size;
	}
	if (len2 + seq1_position_in_genome > genome_size) {
		len2 = genome_size - seq1_position_in_genome;
	}
	//if (len2 <= 0){
        //        return 0.0;
        //}

	char* genome_sub = new char[len2 + 1]; //"TTTGATTGATGGCAGT";//
	for (int i = 0; i < len2; i++) {
		//printf(" i = %d\n", i);
		if (seq1->plus){
			genome_sub[i] = genome[genome_id].data[seq1_position_in_genome + i];
		}
		else{
			genome_sub[i] = genome_minus_sets[genome_id].data[seq1_position_in_genome + i];
		}
	}
	genome_sub[len2] = '\0';
	//printf("seq = %s done\n", seq1->identifier);
	//const char* query = "ACCTCTGCGTGCGATGCTAGCCATCGCTGAC";
	//const char* target = "TCGACACTCTGAAACTCCGTAGCTAGCCTCGATCCTGCACG";
	// default EDLIB_TASK_PATH
	// EDLIB_TASK_DISTANCE
	string seq1_aligned = "";
	string middle = "";
	string seq2_aligned = "";
	string one = "";
	//printf("seq = %s\n", seq1->identifier);
	EdlibAlignResult result = edlibAlign(seq_sub, len1, genome_sub, len2, edlibNewAlignConfig(-1, EDLIB_MODE_HW, EDLIB_TASK_PATH, NULL, 0));
	
	//printf("seq = %s done\n", seq1->identifier);
	if (result.status == EDLIB_STATUS_OK) {
		//printf("%d\n", result.editDistance);
		//printf("%d\n", result.alignmentLength);
		//printf("%d\n", result.endLocations[0]);
		//printAlignment(seq_sub, genome_sub, result.alignment, result.alignmentLength,
		//	*result.endLocations, EDLIB_MODE_HW);
		const unsigned char* alignment = result.alignment;
		const int alignmentLength = result.alignmentLength;
		const int position = *result.endLocations;
		int tIdx = -1;
		int qIdx = -1;
		tIdx = *result.endLocations;
		for (int i = 0; i < alignmentLength; i++) {
			if (alignment[i] != EDLIB_EDOP_INSERT)
				tIdx--;
		}
		for (int start = 0; start < alignmentLength; start += 50) {
			// target
			//printf("T: ");
			int startTIdx;
			for (int j = start; j < start + 50 && j < alignmentLength; j++) {
				if (alignment[j] == EDLIB_EDOP_INSERT)	//printf("-");
					seq2_aligned += "-";
				else
					seq2_aligned += genome_sub[++tIdx];//printf("%c", target[++tIdx]);
				if (j == start)
					startTIdx = tIdx;
			}
			//printf(" (%d - %d)\n", max(startTIdx, 0), tIdx);
			// match / mismatch
			//printf("   ");
			for (int j = start; j < start + 50 && j < alignmentLength; j++) {
				//printf(alignment[j] == EDLIB_EDOP_MATCH ? "|" : " ");
				one = alignment[j] == EDLIB_EDOP_MATCH ? "|" : "*";
				middle += one;
			}
			//printf("\n");

			// query
			//printf("Q: ");
			int startQIdx = qIdx;
			for (int j = start; j < start + 50 && j < alignmentLength; j++) {
				if (alignment[j] == EDLIB_EDOP_DELETE)//printf("-");
					seq1_aligned += "-";
				else
					seq1_aligned += seq_sub[++qIdx]; //printf("%c", query[++qIdx]);
				if (j == start)
					startQIdx = qIdx;
			}
			//printf(" (%d - %d)\n\n", max(startQIdx, 0), qIdx);
		}
	}
	//char* cigar = edlibAlignmentToCigar(result.alignment, result.alignmentLength, EDLIB_CIGAR_STANDARD);
	//printf("%s", cigar);
	//free(cigar);
	edlibFreeAlignResult(result);
	delete[] genome_sub;
	delete[] seq_sub;
	if (seq1_aligned.size() != seq2_aligned.size()) {
		bomb_error("Sequence aligned error: two aligned sequences lengths are not equal!");
	}
	//----------------------------------- backward direction
	string seq1_aligned2 = "";
	string seq2_aligned2 = "";
	string middle2 = "";
	len1 = seq1->position_in_sequence + 6;
	seq_sub = new char[len1 + 1]; //"TTGATTGATGCAGT";// new char[len1];
	for (int i = 1; i <= len1; i++) {
		seq_sub[i - 1] = seq1->data[len1 - i];
	}
	seq_sub[len1] = '\0';
	len2 = ceil(len1 * coefficient) + 5;// 667;
	if (seq1_position_in_genome - len2 < 0){
		len2 = seq1_position_in_genome;
	}
	genome_sub = new char[len2 + 1]; //"TTTGATTGATGGCAGT";//
	for (int i = 0; i < len2; i++) {
		if (seq1->plus){
			genome_sub[i] = genome[genome_id].data[seq1_position_in_genome - i + 5];
		}
		else{
			genome_sub[i] = genome_minus_sets[genome_id].data[seq1_position_in_genome - i + 5];
		}
	}
	genome_sub[len2] = '\0';

	result = edlibAlign(seq_sub, len1, genome_sub, len2, edlibNewAlignConfig(-1, EDLIB_MODE_HW, EDLIB_TASK_PATH, NULL, 0));
	if (result.status == EDLIB_STATUS_OK) {
		//printf("%d\n", result.editDistance);
		//printf("%d\n", result.alignmentLength);
		//printf("%d\n", result.endLocations[0]);
		//printAlignment(seq_sub, genome_sub, result.alignment, result.alignmentLength,
		//*result.endLocations, EDLIB_MODE_HW);
		const unsigned char* alignment = result.alignment;
		const int alignmentLength = result.alignmentLength;
		const int position = *result.endLocations;
		int tIdx = -1;
		int qIdx = -1;
		tIdx = *result.endLocations;
		for (int i = 0; i < alignmentLength; i++) {
			if (alignment[i] != EDLIB_EDOP_INSERT)
				tIdx--;
		}
		for (int start = 0; start < alignmentLength; start += 50) {
			// target
			//printf("T: ");
			int startTIdx;
			for (int j = start; j < start + 50 && j < alignmentLength; j++) {
				if (alignment[j] == EDLIB_EDOP_INSERT)	//printf("-");
					seq2_aligned2 += "-";
				else
					seq2_aligned2 += genome_sub[++tIdx];//printf("%c", target[++tIdx]);
				if (j == start)
					startTIdx = tIdx;
			}
			//printf(" (%d - %d)\n", max(startTIdx, 0), tIdx);
			// match / mismatch
			//printf("   ");
			for (int j = start; j < start + 50 && j < alignmentLength; j++) {
				//printf(alignment[j] == EDLIB_EDOP_MATCH ? "|" : " ");
				one = alignment[j] == EDLIB_EDOP_MATCH ? "|" : "*";
				middle2 += one;
			}
			//printf("\n");

			// query
			//printf("Q: ");
			int startQIdx = qIdx;
			for (int j = start; j < start + 50 && j < alignmentLength; j++) {
				if (alignment[j] == EDLIB_EDOP_DELETE)//printf("-");
					seq1_aligned2 += "-";
				else
					seq1_aligned2 += seq_sub[++qIdx]; //printf("%c", query[++qIdx]);
				if (j == start)
					startQIdx = qIdx;
			}
			//printf(" (%d - %d)\n\n", max(startQIdx, 0), qIdx);
		}
	}
	if (seq1_aligned2.size() != seq2_aligned2.size()) {
		bomb_error("Sequence aligned error: two aligned sequences lengths are not equal!");
	}
	reverse(seq1_aligned2.begin(), seq1_aligned2.end());
	reverse(seq2_aligned2.begin(), seq2_aligned2.end());
	reverse(middle2.begin(), middle2.end());
	int poss = 0;
	int aa = 0;
	for (int w = seq1_aligned2.size(); w >= 0; w--) {
		poss++;
		if (seq1_aligned2[w - 1] == '-') {
			continue;
		}
		else {
			aa++;
		}
		if (aa == 6) {
			break;
		}
	}
	string seq1_whole_aligned = seq1_aligned2.substr(0, seq1_aligned2.size() - poss) + seq1_aligned;
	string seq2_whole_aligned = seq2_aligned2.substr(0, seq2_aligned2.size() - poss) + seq2_aligned;
	string middle_whole_aligned = middle2.substr(0, middle2.size() - poss) + middle;
	int match_num = count(middle_whole_aligned.begin(), middle_whole_aligned.end(), '|');
	int delete_num = count(seq1_whole_aligned.begin(), seq1_whole_aligned.end(), '-');
	int insert_num = count(seq2_whole_aligned.begin(), seq2_whole_aligned.end(), '-');
	int subsitute_num = seq1_whole_aligned.size() - delete_num - insert_num - match_num;
	int aligned_base_num = seq1_whole_aligned.size() - delete_num;
	seq1->Set_aligned_information(seq1_whole_aligned, seq2_whole_aligned, middle_whole_aligned, match_num, subsitute_num, insert_num, delete_num, aligned_base_num, match_num / (float)seq1_whole_aligned.size());
	//char* cigar = edlibAlignmentToCigar(result.alignment, result.alignmentLength, EDLIB_CIGAR_STANDARD);
	//printf("%s", cigar);
	//free(cigar);
	edlibFreeAlignResult(result);
	delete[] genome_sub;
	delete[] seq_sub;
	float lll = (float)seq1_whole_aligned.size();
	return match_num / lll;
}


float GenomeDB::one_seq_align_local_diagonal_banded_for_multi_threads(Sequence *seq1, const Options & options) {
	// store the aligned information, not directly write to a file

	// to avoid storing big whole score matrix
	////////////////////////////// first direction:
	/////////////////////////////  ACCGGTAAAAATCGGCCGAAGCAGAT------>
	/////////////////////////////  ||||||||||||| | || |||||||
	/////////////////////////////  ACCGGTAAAAATCTGGCGGAGCAGAT------>
	if (seq1->plus == -1) {// alignment position not found for too short
		return -0.1;
	}
	int len1 = seq1->size - seq1->position_in_sequence;
	char* seq_sub = new char[len1 + 1]; //"TTGATTGATGCAGT";// new char[len1];
	strcpy(seq_sub, seq1->data + seq1->position_in_sequence);
	seq_sub[len1] = '\0';
	double coefficient = 1.08;// 0 * len2 / len1;
	int len2 = ceil(len1 * coefficient) + 5;// 667;	
	int gapScore = options.gap_score;
	int matchScore = options.match_score;
	int mismatchScore = options.mismatch_score;
	int **matrix; // score matrix
	int **track; // track score
	int *baseIndex; // base index 
	int banded_number_last; // banded/aligned number
	int max_score = 0, max_i = 0, max_j = 0; // store the max score and row, column number
	int max_score2 = 0, max_i2 = 0, max_j2 = 0; // store the max score and row, column number at the len1-th matrixScore
	int max_score_final = 0, max_i_final = 0, max_j_final = 0;
	string seq1_aligned = "";
	string seq2_aligned = "";
	string middle = "";
	string direction = "";
	int from_up, from_diag, from_left, index_in_sequence, index_in_genome, i, j;
	int match_num = 0, delete_num = 0, insert_num = 0, subsitute_num = 0, aligned_base_num = 0;
	int banded_up, banded_down, banded_width;// , match_number = 0;
	if (len1 < 8000) {
		banded_width = ceil(len1 / 5.0);
	}
	else {
		banded_width = ceil((len2 - len1) * 1.3);
	}
	//printf("len1 = %d, band width = %d\n", len1, banded_width);
	int seq1_position_in_genome = 0;
	Sequence *genome_is;
	if (seq1->plus) {
		genome_is = genome;
		seq1_position_in_genome = seq1->position_in_genome;
		direction = "Plus";
	}
	else {
		genome_is = genome_minus;
		seq1_position_in_genome = seq1->position_in_genome_minus;
		direction = "Minus";
	}
	char* genome_sub = new char[len2 + 1]; //"TTTGATTGATGGCAGT";//
	for (int i = 0; i < len2; i++) {
		genome_sub[i] = genome_is->data[seq1_position_in_genome + i];
	}
	genome_sub[len2] = '\0';
	//cout << genome_sub << endl;

	//  up 1, diag 2, left 3
	matrix = new int*[len1 + 1]; //设置行 或直接double **data=new double*[m]; 一个指针指向一个指针数组。  
	track = new int*[len1 + 1];
	baseIndex = new int[len1 + 1]();
	for (int j = 0; j <= len1; j++) {
		//printf("j = %d\n", j);
		matrix[j] = new int[2 * banded_width + 5]();    // initialize each value to 0 这个指针数组的每个指针元素又指向一个数组
		track[j] = new int[2 * banded_width + 5]();
	}
	for (i = 1; i <= len1; i++) {
		index_in_sequence = i - 1;///seq1->position_in_sequence + i;
		banded_up = min(len2, ceil(coefficient * i + banded_width));
		banded_down = max(1, floor(coefficient * i - banded_width));
		baseIndex[i] = max(0, banded_down);
		for (j = banded_down; j <= banded_up; j++) {
			index_in_genome = j - 1;
			//cout << "seq[" << index_in_sequence - 1 << "] = " << seq1->data[index_in_sequence - 1] << endl;
			//cout << "gen[" << index_in_genome - 1 << "} = " << genome_is->data[index_in_genome - 1] << "---------" <<endl;
			if (seq_sub[index_in_sequence] == genome_sub[index_in_genome]) {
				if (j - 1 - baseIndex[i - 1] >= 0) {
					from_diag = matrix[i - 1][j - 1 - baseIndex[i - 1]] + matchScore;
				}
				else {
					from_diag = 2;
				}

			}
			else {
				if (j - 1 - baseIndex[i - 1] >= 0) {
					from_diag = matrix[i - 1][j - 1 - baseIndex[i - 1]] + mismatchScore;
				}
				else {
					from_diag = -2;
				}

			}
			from_up = matrix[i - 1][j - baseIndex[i - 1]] + gapScore;
			from_left = max(0, matrix[i][j - 1 - baseIndex[i]] + gapScore);
			if (from_diag >= from_up && from_diag >= from_left && from_diag > 0) {
				matrix[i][j - baseIndex[i]] = from_diag;
				track[i][j - baseIndex[i]] = 2;//  up 1, diag 2, left 3
				if (from_diag > max_score) {
					max_score = from_diag;
					max_i = i;
					max_j = j;
				}
			}

			else if (from_up >= from_diag && from_up >= from_left && from_up > 0) {
				matrix[i][j - baseIndex[i]] = from_up;
				track[i][j - baseIndex[i]] = 1;//  up 1, diag 2, left 3
				if (from_up > max_score) {
					max_score = from_up;
					max_i = i;
					max_j = j;
				}
			}
			else if (from_left >= from_diag && from_left >= from_up && from_left > 0) {
				matrix[i][j - baseIndex[i]] = from_left;
				track[i][j - baseIndex[i]] = 3;//  up 1, diag 2, left 3
				if (from_left > max_score) {
					max_score = from_left;
					max_i = i;
					max_j = j;
				}
			}
			else {
				matrix[i][j - baseIndex[i]] = 0;
				track[i][j - baseIndex[i]] = 0;
			}
			//cout << "M[" << i << "][" << j << "] = " << matrix[i][j] << endl;

		}
		//printf("max_score = %d, i = %d, j = %d\n", max_score, max_i, max_j);
	}
	banded_number_last = banded_up - banded_down;
	if (max_i < len1) {
		for (int jjjj = 0; jjjj < banded_number_last; jjjj++) {
			if (matrix[len1][jjjj] > max_score2) {
				max_score2 = matrix[len1][jjjj];
				max_j2 = jjjj + baseIndex[len1];
			}
		}
		if (max_score2 > 0) {
			max_score_final = max_score2;
			max_i_final = len1;
			max_j_final = max_j2;
		}
		else {
			max_score_final = max_score;
			max_i_final = max_i;
			max_j_final = max_j;
		}
	}
	else {
		max_i_final = max_i;
		max_j_final = max_j;
	}
	i = max_i_final;// len1 - 1;
	j = max_j_final;// len2 - 1;
	for (; i > 0 && j > 0;) {
		//index_in_sequence = i - 1;
		//index_in_genome = j - 1;
		if (track[i][j - baseIndex[i]] == 2) {//  up 1, diag 2, left 3
			seq1_aligned = seq_sub[i - 1] + seq1_aligned;
			seq2_aligned = genome_sub[j - 1] + seq2_aligned;
			if (seq_sub[i - 1] == genome_sub[j - 1]) {
				middle = "|" + middle;
			}
			else
				middle = " " + middle;
			i--;
			j--;
		}
		else if (track[i][j - baseIndex[i]] == 1) {//  up 1, diag 2, left 3
			seq1_aligned = seq_sub[i - 1] + seq1_aligned;
			i--;
			seq2_aligned = "-" + seq2_aligned;//haha.at((int)seq2->data[j]);
			middle = " " + middle;
		}
		else {
			seq1_aligned = "-" + seq1_aligned;//haha.at((int)seq1->data[i - 1]);
			seq2_aligned = genome_sub[j - 1] + seq2_aligned;
			j--;
			middle = " " + middle;
		}
	}
	if (i > 0) {
		for (; i > 0; i--) {
			seq1_aligned = seq_sub[i - 1] + seq1_aligned;
			seq2_aligned = "-" + seq2_aligned;
			middle = " " + middle;
		}
	}
	if (seq1_aligned.size() != seq2_aligned.size()) {
		bomb_error("Sequence aligned error: two aligned sequences lengths are not equal!");
	}
#if 0
	ofstream outfile;
	outfile.open("D:\\OTU拼接\\回国后文章\\c++代码\\out.txt", ios::out);
	if (!outfile.is_open())
		bomb_error("Open file failure");
	//for (int aaa = 0; aaa < location_num; aaa++) {
	outfile << seq1_aligned << endl;
	outfile << middle << endl;
	outfile << seq2_aligned << endl;
	//outfile << array_location[aaa].index_in_sequence << "\t" << array_location[aaa].index_in_genome << endl;  //在result.txt中写入结果
	//}
	outfile.close();
#endif
	// release memory
	for (int i = 0; i <= len1; i++) {
		delete[] matrix[i];
		delete[] track[i];
	}
	delete[] matrix;
	delete[] track;
	delete[] baseIndex;
	delete[] genome_sub;
	delete[] seq_sub;
	int aaaa = 0;
	//--------------------------------------------------------------------------------------
	////////////////////////////// second direction:
	/////////////////////////////  <-------ACCGGTAAAAA  TCGGCCGAAGCAGAT
	/////////////////////////////          |||||||||||  || | || |||||||
	/////////////////////////////  <-------ACCGGTAAAAA  TCTGGCGGAGCAGAT
	int **matrix2; // score matrix
	int **track2; // track score
	int *baseIndex2; // base index 
	max_score = 0, max_i = 0, max_j = 0; // store the max score and row, column number
	max_score2 = 0, max_i2 = 0, max_j2 = 0; // store the max score and row, column number at the len1-th matrixScore
	max_score_final = 0, max_i_final = 0, max_j_final = 0;
	string seq1_aligned2 = "";
	string seq2_aligned2 = "";
	string middle2 = "";
	//int match_num2 = 0, delete_num2 = 0, insert_num2 = 0;
	len1 = seq1->position_in_sequence + 6;
	seq_sub = new char[len1 + 1]; //"TTGATTGATGCAGT";// new char[len1];
	for (int i = 1; i <= len1; i++) {
		seq_sub[i - 1] = seq1->data[len1 - i];
	}
	seq_sub[len1] = '\0';
	len2 = ceil(len1 * coefficient) + 5;// 667;
	genome_sub = new char[len2 + 1]; //"TTTGATTGATGGCAGT";//
	for (int i = 0; i < len2; i++) {
		genome_sub[i] = genome_is->data[seq1_position_in_genome - i + 5];
	}
	genome_sub[len2] = '\0';
	if (len1 < 8000) {
		banded_width = ceil(len1 / 5.0);
	}
	else {
		banded_width = ceil((len2 - len1) * 1.3);
	}
	//banded_width = max(ceil(len1/2.0), ceil((len2 - len1) * 1.3));
	//cout << genome_sub << endl;
	//printf("len1 = %d, band width = %d\n", len1, banded_width);

	//  up 1, diag 2, left 3
	matrix2 = new int*[len1 + 1]; //设置行 或直接double **data=new double*[m]; 一个指针指向一个指针数组。  
	track2 = new int*[len1 + 1];
	baseIndex2 = new int[len1 + 1]();
	for (int j = 0; j <= len1; j++) {
		//printf("j = %d\n", j);
		matrix2[j] = new int[2 * banded_width + 5]();    // initialize each value to 0 这个指针数组的每个指针元素又指向一个数组
		track2[j] = new int[2 * banded_width + 5]();
	}
	for (i = 1; i <= len1; i++) {
		index_in_sequence = i - 1;///seq1->position_in_sequence + i;
		banded_up = min(len2, ceil(coefficient * i + banded_width));
		banded_down = max(1, floor(coefficient * i - banded_width));
		baseIndex2[i] = max(0, banded_down);
		for (j = banded_down; j <= banded_up; j++) {
			index_in_genome = j - 1;
			//cout << "seq[" << index_in_sequence - 1 << "] = " << seq1->data[index_in_sequence - 1] << endl;
			//cout << "gen[" << index_in_genome - 1 << "} = " << genome_is->data[index_in_genome - 1] << "---------" <<endl;
			if (seq_sub[index_in_sequence] == genome_sub[index_in_genome]) {
				if (j - 1 - baseIndex2[i - 1] >= 0) {
					from_diag = matrix2[i - 1][j - 1 - baseIndex2[i - 1]] + matchScore;
				}
				else {
					from_diag = 2;
				}

			}
			else {
				if (j - 1 - baseIndex2[i - 1] >= 0) {
					from_diag = matrix2[i - 1][j - 1 - baseIndex2[i - 1]] + mismatchScore;
				}
				else {
					from_diag = -2;
				}

			}
			from_up = matrix2[i - 1][j - baseIndex2[i - 1]] + gapScore;
			from_left = max(0, matrix2[i][j - 1 - baseIndex2[i]] + gapScore);
			if (from_diag >= from_up && from_diag >= from_left && from_diag > 0) {
				matrix2[i][j - baseIndex2[i]] = from_diag;
				track2[i][j - baseIndex2[i]] = 2;//  up 1, diag 2, left 3
				if (from_diag > max_score) {
					max_score = from_diag;
					max_i = i;
					max_j = j;
				}
			}

			else if (from_up >= from_diag && from_up >= from_left && from_up > 0) {
				matrix2[i][j - baseIndex2[i]] = from_up;
				track2[i][j - baseIndex2[i]] = 1;//  up 1, diag 2, left 3
				if (from_up > max_score) {
					max_score = from_up;
					max_i = i;
					max_j = j;
				}
			}
			else if (from_left >= from_diag && from_left >= from_up && from_left > 0) {
				matrix2[i][j - baseIndex2[i]] = from_left;
				track2[i][j - baseIndex2[i]] = 3;//  up 1, diag 2, left 3
				if (from_left > max_score) {
					max_score = from_left;
					max_i = i;
					max_j = j;
				}
			}
			else {
				matrix2[i][j - baseIndex2[i]] = 0;
				track2[i][j - baseIndex2[i]] = 0;
			}
			//cout << "M[" << i << "][" << j << "] = " << matrix[i][j] << endl;

		}
		//printf("max_score = %d, i = %d, j = %d\n", max_score, max_i, max_j);
	}
	banded_number_last = banded_up - banded_down;
	if (max_i < len1) {
		for (int jjjj = 0; jjjj < banded_number_last; jjjj++) {
			if (matrix2[len1][jjjj] > max_score2) {
				max_score2 = matrix2[len1][jjjj];
				max_j2 = jjjj + baseIndex2[len1];
			}
		}
		if (max_score2 > 0) {
			max_score_final = max_score2;
			max_i_final = len1;
			max_j_final = max_j2;
		}
		else {
			max_score_final = max_score;
			max_i_final = max_i;
			max_j_final = max_j;
		}
	}
	else {
		max_i_final = max_i;
		max_j_final = max_j;
	}
	i = max_i_final;// len1 - 1;
	j = max_j_final;// len2 - 1;
	for (; i > 0 && j > 0;) {
		//index_in_sequence = i - 1;
		//index_in_genome = j - 1;
		if (track2[i][j - baseIndex2[i]] == 2) {//  up 1, diag 2, left 3
			seq1_aligned2 = seq_sub[i - 1] + seq1_aligned2;
			seq2_aligned2 = genome_sub[j - 1] + seq2_aligned2;
			if (seq_sub[i - 1] == genome_sub[j - 1]) {
				middle2 = "|" + middle2;
			}
			else
				middle2 = " " + middle2;
			i--;
			j--;
		}
		else if (track2[i][j - baseIndex2[i]] == 1) {//  up 1, diag 2, left 3
			seq1_aligned2 = seq_sub[i - 1] + seq1_aligned2;
			i--;
			seq2_aligned2 = "-" + seq2_aligned2;//haha.at((int)seq2->data[j]);
			middle2 = " " + middle2;
		}
		else {
			seq1_aligned2 = "-" + seq1_aligned2;//haha.at((int)seq1->data[i - 1]);
			seq2_aligned2 = genome_sub[j - 1] + seq2_aligned2;
			j--;
			middle2 = " " + middle2;
		}
	}
	if (i > 0) {
		for (; i > 0; i--) {
			seq1_aligned2 = seq_sub[i - 1] + seq1_aligned2;
			seq2_aligned2 = "-" + seq2_aligned2;
			middle2 = " " + middle2;
		}
	}
	if (seq1_aligned2.size() != seq2_aligned2.size()) {
		bomb_error("Sequence aligned error: two aligned sequences lengths are not equal!");
	}
	reverse(seq1_aligned2.begin(), seq1_aligned2.end());
	reverse(seq2_aligned2.begin(), seq2_aligned2.end());
	reverse(middle2.begin(), middle2.end());
	int poss = 0;
	int aa = 0;
	for (int w = seq1_aligned2.size(); w >= 0; w--) {
		poss++;
		if (seq1_aligned2[w - 1] == '-') {
			continue;
		}
		else {
			aa++;
		}
		if (aa == 6) {
			break;
		}
	}
	string seq1_whole_aligned = seq1_aligned2.substr(0, seq1_aligned2.size() - poss) + seq1_aligned;
	string seq2_whole_aligned = seq2_aligned2.substr(0, seq2_aligned2.size() - poss) + seq2_aligned;
	string middle_whole_aligned = middle2.substr(0, middle2.size() - poss) + middle;
	match_num = count(middle_whole_aligned.begin(), middle_whole_aligned.end(), '|');
	delete_num = count(seq1_whole_aligned.begin(), seq1_whole_aligned.end(), '-');
	insert_num = count(seq2_whole_aligned.begin(), seq2_whole_aligned.end(), '-');
	subsitute_num = seq1_whole_aligned.size() - delete_num - insert_num - match_num;
	aligned_base_num = seq1_whole_aligned.size() - delete_num;
	seq1->Set_aligned_information(seq1_whole_aligned, seq2_whole_aligned, middle_whole_aligned, match_num, subsitute_num, insert_num, delete_num, aligned_base_num, match_num / (float)seq1_whole_aligned.size());
	ofstream outfile2;
	string ff = "test.txt";
	const char* filename = ff.data();
	outfile2.open(filename, ios::out);
	//outfile2.open(options.output, ios::app);
	if (!outfile2.is_open())
		bomb_error("Open output file failure");
	//for (int aaa = 0; aaa < location_num; aaa++) {
	outfile2 << seq1->identifier << endl;
	outfile2 << seq1->data << endl;
	outfile2 << seq1_whole_aligned << endl;
	outfile2 << middle_whole_aligned << endl;
	outfile2 << seq2_whole_aligned << endl;
	outfile2.close();
	for (int i = 0; i <= len1; i++) {
		delete[] matrix2[i];
		delete[] track2[i];
	}
	delete[] matrix2;
	delete[] track2;
	delete[] baseIndex2;
	delete[] genome_sub;
	delete[] seq_sub;
	return match_num / (float)seq1_whole_aligned.size();
}

float GenomeDB::seq_align_local_diagonal_banded_one_matrix(Sequence *seq1, const Options & options, int **matrix, int **track) {
	// to avoid storing big whole score matrix
	////////////////////////////// first direction:
	/////////////////////////////  ACCGGTAAAAATCGGCCGAAGCAGAT------>
	/////////////////////////////  ||||||||||||| | || |||||||
	/////////////////////////////  ACCGGTAAAAATCTGGCGGAGCAGAT------>
	int len1 = seq1->size - seq1->position_in_sequence;
	char* seq_sub = new char[len1 + 1]; //"TTGATTGATGCAGT";// new char[len1];
	strcpy(seq_sub, seq1->data + seq1->position_in_sequence);
	seq_sub[len1] = '\0';
	double coefficient = 1.08;// 0 * len2 / len1;
	int len2 = ceil(len1 * coefficient) + 5;// 667;	
	int gapScore = options.gap_score;
	int matchScore = options.match_score;
	int mismatchScore = options.mismatch_score;
	//int **matrix; // score matrix
	//int **track; // track score
	int *baseIndex; // base index 
	int banded_number_last; // banded/aligned number
	int max_score = 0, max_i = 0, max_j = 0; // store the max score and row, column number
	int max_score2 = 0, max_i2 = 0, max_j2 = 0; // store the max score and row, column number at the len1-th matrixScore
	int max_score_final = 0, max_i_final = 0, max_j_final = 0;
	string seq1_aligned = "";
	string seq2_aligned = "";
	string middle = "";
	string direction = "";
	int from_up, from_diag, from_left, index_in_sequence, index_in_genome, i, j;
	int match_num = 0, delete_num = 0, insert_num = 0, subsitute_num = 0;
	int banded_up, banded_down, banded_width;// , match_number = 0;
	if (len1 < 8000) {
		banded_width = ceil(len1 / 3.0);
	}
	else {
		//banded_width = ceil(len1 * 0.105);//ceil((len2 - len1) * 1.3);
		banded_width = min(options.max_banded_width, ceil(len1 * 0.205));
	}
	//printf("len1 = %d, band width = %d\n", len1, banded_width);
	int seq1_position_in_genome = 0;
	Sequence *genome_is;
	if (seq1->plus) {
		genome_is = genome;
		seq1_position_in_genome = seq1->position_in_genome;
		direction = "Plus";
	}
	else {
		genome_is = genome_minus;
		seq1_position_in_genome = seq1->position_in_genome_minus;
		direction = "Minus";
	}
	char* genome_sub = new char[len2 + 1]; //"TTTGATTGATGGCAGT";//
	for (int i = 0; i < len2; i++) {
		genome_sub[i] = genome_is->data[seq1_position_in_genome + i];
	}
	genome_sub[len2] = '\0';
	//cout << genome_sub << endl;

	//  up 1, diag 2, left 3
	//matrix = new int*[len1 + 1]; //设置行 或直接double **data=new double*[m]; 一个指针指向一个指针数组。  
	//track = new int*[len1 + 1];
	baseIndex = new int[len1 + 1]();
	for (int ii = 0; ii <= len1; ii++) {
		for (int jj = 0; jj <= 2 * banded_width; jj++) {
			matrix[ii][jj] = 0;
			track[ii][jj] = 0;
		}
	}
	//	//printf("j = %d\n", j);
	//	matrix[j] = new int[2 * banded_width + 5]();    // initialize each value to 0 这个指针数组的每个指针元素又指向一个数组
	//	track[j] = new int[2 * banded_width + 5]();
	//}
	//printf("enter\n");

	for (i = 1; i <= len1; i++) {
		index_in_sequence = i - 1;///seq1->position_in_sequence + i;
		banded_up = min(len2, ceil(coefficient * i + banded_width));
		banded_down = max(1, floor(coefficient * i - banded_width));
		baseIndex[i] = max(0, banded_down);
		for (j = banded_down; j <= banded_up; j++) {
			index_in_genome = j - 1;
			//cout << "seq[" << index_in_sequence - 1 << "] = " << seq1->data[index_in_sequence - 1] << endl;
			//cout << "gen[" << index_in_genome - 1 << "} = " << genome_is->data[index_in_genome - 1] << "---------" <<endl;
			if (seq_sub[index_in_sequence] == genome_sub[index_in_genome]) {
				if (j - 1 - baseIndex[i - 1] >= 0) {
					from_diag = matrix[i - 1][j - 1 - baseIndex[i - 1]] + matchScore;
				}
				else {
					from_diag = 2;
				}

			}
			else {
				if (j - 1 - baseIndex[i - 1] >= 0) {
					from_diag = matrix[i - 1][j - 1 - baseIndex[i - 1]] + mismatchScore;
				}
				else {
					from_diag = -2;
				}

			}
			from_up = matrix[i - 1][j - baseIndex[i - 1]] + gapScore;
			from_left = max(0, matrix[i][j - 1 - baseIndex[i]] + gapScore);
			if (from_diag >= from_up && from_diag >= from_left && from_diag > 0) {
				matrix[i][j - baseIndex[i]] = from_diag;
				track[i][j - baseIndex[i]] = 2;//  up 1, diag 2, left 3
				if (from_diag > max_score) {
					max_score = from_diag;
					max_i = i;
					max_j = j;
				}
			}

			else if (from_up >= from_diag && from_up >= from_left && from_up > 0) {
				matrix[i][j - baseIndex[i]] = from_up;
				track[i][j - baseIndex[i]] = 1;//  up 1, diag 2, left 3
				if (from_up > max_score) {
					max_score = from_up;
					max_i = i;
					max_j = j;
				}
			}
			else if (from_left >= from_diag && from_left >= from_up && from_left > 0) {
				matrix[i][j - baseIndex[i]] = from_left;
				track[i][j - baseIndex[i]] = 3;//  up 1, diag 2, left 3
				if (from_left > max_score) {
					max_score = from_left;
					max_i = i;
					max_j = j;
				}
			}
			else {
				matrix[i][j - baseIndex[i]] = 0;
				track[i][j - baseIndex[i]] = 0;
			}
			//cout << "M[" << i << "][" << j << "] = " << matrix[i][j] << endl;

		}
		//printf("max_score = %d, i = %d, j = %d\n", max_score, max_i, max_j);
	}
	banded_number_last = banded_up - banded_down;
	//printf("max_score = %d  ", max_score);
	if (max_i < len1) {
		for (int jjjj = 0; jjjj < banded_number_last; jjjj++) {
			if (matrix[len1][jjjj] > max_score2) {
				max_score2 = matrix[len1][jjjj];
				max_j2 = jjjj + baseIndex[len1];
			}
		}
		if (max_score2 > 0) {
			max_score_final = max_score2;
			max_i_final = len1;
			max_j_final = max_j2;
		}
		else {
			max_score_final = max_score;
			max_i_final = max_i;
			max_j_final = max_j;
		}
	}
	else {
		max_i_final = max_i;
		max_j_final = max_j;
	}
	i = max_i_final;// len1 - 1;
	j = max_j_final;// len2 - 1;
	for (; i > 0 && j > 0;) {
		//index_in_sequence = i - 1;
		//index_in_genome = j - 1;
		if (track[i][j - baseIndex[i]] == 2) {//  up 1, diag 2, left 3
			seq1_aligned = seq_sub[i - 1] + seq1_aligned;
			seq2_aligned = genome_sub[j - 1] + seq2_aligned;
			if (seq_sub[i - 1] == genome_sub[j - 1]) {
				middle = "|" + middle;
			}
			else
				middle = " " + middle;
			i--;
			j--;
		}
		else if (track[i][j - baseIndex[i]] == 1) {//  up 1, diag 2, left 3
			seq1_aligned = seq_sub[i - 1] + seq1_aligned;
			i--;
			seq2_aligned = "-" + seq2_aligned;//haha.at((int)seq2->data[j]);
			middle = " " + middle;
		}
		else {
			seq1_aligned = "-" + seq1_aligned;//haha.at((int)seq1->data[i - 1]);
			seq2_aligned = genome_sub[j - 1] + seq2_aligned;
			j--;
			middle = " " + middle;
		}
	}
	if (seq1_aligned.size() != seq2_aligned.size()) {
		bomb_error("Sequence aligned error: two aligned sequences lengths are not equal!");
	}
#if 0
	ofstream outfile;
	outfile.open("D:\\OTU拼接\\回国后文章\\c++代码\\out.txt", ios::out);
	if (!outfile.is_open())
		bomb_error("Open file failure");
	//for (int aaa = 0; aaa < location_num; aaa++) {
	outfile << seq1_aligned << endl;
	outfile << middle << endl;
	outfile << seq2_aligned << endl;
	//outfile << array_location[aaa].index_in_sequence << "\t" << array_location[aaa].index_in_genome << endl;  //在result.txt中写入结果
	//}
	outfile.close();
#endif
	// release memory
	//for (int i = 0; i <= len1; i++) {
	//	delete[] matrix[i];
	//	delete[] track[i];
	//}
	//delete[] matrix;
	//delete[] track;
	delete[] baseIndex;
	delete[] genome_sub;
	delete[] seq_sub;
	int aaaa = 0;
	//--------------------------------------------------------------------------------------
	////////////////////////////// second direction:
	/////////////////////////////  <-------ACCGGTAAAAA  TCGGCCGAAGCAGAT
	/////////////////////////////          |||||||||||  || | || |||||||
	/////////////////////////////  <-------ACCGGTAAAAA  TCTGGCGGAGCAGAT
	//int **matrix2; // score matrix
	//int **track2; // track score
	int *baseIndex2; // base index 
	max_score = 0, max_i = 0, max_j = 0; // store the max score and row, column number
	max_score2 = 0, max_i2 = 0, max_j2 = 0; // store the max score and row, column number at the len1-th matrixScore
	max_score_final = 0, max_i_final = 0, max_j_final = 0;
	string seq1_aligned2 = "";
	string seq2_aligned2 = "";
	string middle2 = "";
	//int match_num2 = 0, delete_num2 = 0, insert_num2 = 0;
	len1 = seq1->position_in_sequence + 6;
	seq_sub = new char[len1]; //"TTGATTGATGCAGT";// new char[len1];
	for (int i = 1; i <= len1; i++) {
		seq_sub[i - 1] = seq1->data[len1 - i];
	}
	len2 = ceil(len1 * coefficient) + 5;// 667;
	genome_sub = new char[len2 + 1]; //"TTTGATTGATGGCAGT";//
	for (int i = 0; i < len2; i++) {
		genome_sub[i] = genome_is->data[seq1_position_in_genome - i + 5];
	}
	genome_sub[len2] = '\0';
	if (len1 < 8000) {
		banded_width = ceil(len1 / 3.0);
	}
	else {
		banded_width = min(options.max_banded_width, ceil(len1 * 0.205));
	}
	//banded_width = max(ceil(len1/2.0), ceil((len2 - len1) * 1.3));
	//cout << genome_sub << endl;
	//printf("len1 = %d, band width = %d\n", len1, banded_width);

	//  up 1, diag 2, left 3
	//matrix2 = new int*[len1 + 1]; //设置行 或直接double **data=new double*[m]; 一个指针指向一个指针数组。  
	//track2 = new int*[len1 + 1];
	baseIndex2 = new int[len1 + 1]();
	for (int ii = 0; ii <= len1; ii++) {
		for (int jj = 0; jj <= 2 * banded_width; jj++) {
			matrix[ii][jj] = 0;
			track[ii][jj] = 0;
		}
	}

	//for (int j = 0; j <= len1; j++) {
	//printf("j = %d\n", j);
	//	matrix2[j] = new int[2 * banded_width + 5]();    // initialize each value to 0 这个指针数组的每个指针元素又指向一个数组
	//	track2[j] = new int[2 * banded_width + 5]();
	//}
	for (i = 1; i <= len1; i++) {
		index_in_sequence = i - 1;///seq1->position_in_sequence + i;
		banded_up = min(len2, ceil(coefficient * i + banded_width));
		banded_down = max(1, floor(coefficient * i - banded_width));
		baseIndex2[i] = max(0, banded_down);
		for (j = banded_down; j <= banded_up; j++) {
			index_in_genome = j - 1;
			//cout << "seq[" << index_in_sequence - 1 << "] = " << seq1->data[index_in_sequence - 1] << endl;
			//cout << "gen[" << index_in_genome - 1 << "} = " << genome_is->data[index_in_genome - 1] << "---------" <<endl;
			if (seq_sub[index_in_sequence] == genome_sub[index_in_genome]) {
				if (j - 1 - baseIndex2[i - 1] >= 0) {
					from_diag = matrix[i - 1][j - 1 - baseIndex2[i - 1]] + matchScore;
				}
				else {
					from_diag = 2;
				}

			}
			else {
				if (j - 1 - baseIndex2[i - 1] >= 0) {
					from_diag = matrix[i - 1][j - 1 - baseIndex2[i - 1]] + mismatchScore;
				}
				else {
					from_diag = -2;
				}

			}
			from_up = matrix[i - 1][j - baseIndex2[i - 1]] + gapScore;
			from_left = max(0, matrix[i][j - 1 - baseIndex2[i]] + gapScore);
			if (from_diag >= from_up && from_diag >= from_left && from_diag > 0) {
				matrix[i][j - baseIndex2[i]] = from_diag;
				track[i][j - baseIndex2[i]] = 2;//  up 1, diag 2, left 3
				if (from_diag > max_score) {
					max_score = from_diag;
					max_i = i;
					max_j = j;
				}
			}

			else if (from_up >= from_diag && from_up >= from_left && from_up > 0) {
				matrix[i][j - baseIndex2[i]] = from_up;
				track[i][j - baseIndex2[i]] = 1;//  up 1, diag 2, left 3
				if (from_up > max_score) {
					max_score = from_up;
					max_i = i;
					max_j = j;
				}
			}
			else if (from_left >= from_diag && from_left >= from_up && from_left > 0) {
				matrix[i][j - baseIndex2[i]] = from_left;
				track[i][j - baseIndex2[i]] = 3;//  up 1, diag 2, left 3
				if (from_left > max_score) {
					max_score = from_left;
					max_i = i;
					max_j = j;
				}
			}
			else {
				matrix[i][j - baseIndex2[i]] = 0;
				track[i][j - baseIndex2[i]] = 0;
			}
			//cout << "M[" << i << "][" << j << "] = " << matrix[i][j] << endl;

		}
		//printf("max_score2 = %d, i = %d, j = %d\n", max_score, max_i, max_j);
	}
	banded_number_last = banded_up - banded_down;
	//printf("maxscore2 = %d\n", max_score);
	if (max_i < len1) {
		for (int jjjj = 0; jjjj < banded_number_last; jjjj++) {
			if (matrix[len1][jjjj] > max_score2) {
				max_score2 = matrix[len1][jjjj];
				max_j2 = jjjj + baseIndex2[len1];
			}
		}
		if (max_score2 > 0) {
			max_score_final = max_score2;
			max_i_final = len1;
			max_j_final = max_j2;
		}
		else {
			max_score_final = max_score;
			max_i_final = max_i;
			max_j_final = max_j;
		}
	}
	else {
		max_i_final = max_i;
		max_j_final = max_j;
	}
	i = max_i_final;// len1 - 1;
	j = max_j_final;// len2 - 1;
					//printf("final_i = %d, len1 = %d\n", i, len1);
	for (; i > 0 && j > 0;) {
		//index_in_sequence = i - 1;
		//index_in_genome = j - 1;
		if (track[i][j - baseIndex2[i]] == 2) {//  up 1, diag 2, left 3
			seq1_aligned2 = seq_sub[i - 1] + seq1_aligned2;
			seq2_aligned2 = genome_sub[j - 1] + seq2_aligned2;
			if (seq_sub[i - 1] == genome_sub[j - 1]) {
				middle2 = "|" + middle2;
			}
			else
				middle2 = " " + middle2;
			i--;
			j--;
		}
		else if (track[i][j - baseIndex2[i]] == 1) {//  up 1, diag 2, left 3
			seq1_aligned2 = seq_sub[i - 1] + seq1_aligned2;
			i--;
			seq2_aligned2 = "-" + seq2_aligned2;//haha.at((int)seq2->data[j]);
			middle2 = " " + middle2;
		}
		else {
			seq1_aligned2 = "-" + seq1_aligned2;//haha.at((int)seq1->data[i - 1]);
			seq2_aligned2 = genome_sub[j - 1] + seq2_aligned2;
			j--;
			middle2 = " " + middle2;
		}
	}
	if (seq1_aligned2.size() != seq2_aligned2.size()) {
		bomb_error("Sequence aligned error: two aligned sequences lengths are not equal!");
	}
	reverse(seq1_aligned2.begin(), seq1_aligned2.end());
	reverse(seq2_aligned2.begin(), seq2_aligned2.end());
	reverse(middle2.begin(), middle2.end());
	string seq1_whole_aligned = seq1_aligned2.substr(0, seq1_aligned2.size() - 6) + seq1_aligned;
	string seq2_whole_aligned = seq2_aligned2.substr(0, seq2_aligned2.size() - 6) + seq2_aligned;
	string middle_whole_aligned = middle2.substr(0, middle2.size() - 6) + middle;
	match_num = count(middle_whole_aligned.begin(), middle_whole_aligned.end(), '|');
	delete_num = count(seq1_whole_aligned.begin(), seq1_whole_aligned.end(), '-');
	insert_num = count(seq2_whole_aligned.begin(), seq2_whole_aligned.end(), '-');
	subsitute_num = seq1_whole_aligned.size() - delete_num - insert_num - match_num;
	//printf("aligned length = %d\n", seq1_whole_aligned.size());
	ofstream outfile2;
	const char* filename = options.output.data();
	outfile2.open(filename, ios::app);
	//outfile2.open(options.output, ios::app);
	if (!outfile2.is_open())
		bomb_error("Open output file failure");
	//for (int aaa = 0; aaa < location_num; aaa++) {
	outfile2 << "Query:      " << seq1->identifier << endl;
	outfile2 << "Length:     " << seq1->size << endl;
	outfile2 << "nMatch:     " << match_num << endl;
	outfile2 << "nSubsitute: " << subsitute_num << endl;
	outfile2 << "nDelete:    " << delete_num << endl;
	outfile2 << "nInsert:    " << insert_num << endl;
	outfile2 << "Identify:   " << match_num / (float)seq1_whole_aligned.size() << endl;
	//outfile2 << "Score:      " << insert_num << endl;
	outfile2 << "Strand:     Plus/" << direction << endl;
	outfile2 << endl;
	int leftPosition = 0, rightPosition = 0;
	for (int iii = 0; iii < seq1_whole_aligned.size(); ) {
		if (leftPosition >= seq1_whole_aligned.size()) {
			break;
		}
		outfile2 << "Query  " << seq1_whole_aligned.substr(leftPosition, 60) << endl;
		outfile2 << "       " << middle_whole_aligned.substr(leftPosition, 60) << endl;
		outfile2 << "Sbjct  " << seq2_whole_aligned.substr(leftPosition, 60) << endl;
		outfile2 << endl;
		leftPosition += 60;
	}
	outfile2 << endl;
	outfile2 << endl;
	//outfile2 << seq1_whole_aligned << endl;
	//outfile2 << middle_whole_aligned << endl;
	//outfile2 << seq2_whole_aligned << endl;
	//outfile << array_location[aaa].index_in_sequence << "\t" << array_location[aaa].index_in_genome << endl;  //在result.txt中写入结果
	//}
	outfile2.close();
	//#endif
	// release memory
	//for (int i = 0; i <= len1; i++) {
	//	delete[] matrix2[i];
	//	delete[] track2[i];
	//}
	//delete[] matrix2;
	//delete[] track2;
	delete[] baseIndex2;
	delete[] genome_sub;
	delete[] seq_sub;
	return match_num / (float)seq1_whole_aligned.size();
}

void printAlignment(const char* query, const char* target,
	const unsigned char* alignment, const int alignmentLength,
	const int position, const EdlibAlignMode modeCode) {
	int tIdx = -1;
	int qIdx = -1;
	if (modeCode == EDLIB_MODE_HW) {
		tIdx = position;
		for (int i = 0; i < alignmentLength; i++) {
			if (alignment[i] != EDLIB_EDOP_INSERT)
				tIdx--;
		}
	}
	for (int start = 0; start < alignmentLength; start += 50) {
		// target
		printf("T: ");
		int startTIdx;
		for (int j = start; j < start + 50 && j < alignmentLength; j++) {
			if (alignment[j] == EDLIB_EDOP_INSERT)
				printf("-");
			else
				printf("%c", target[++tIdx]);
			if (j == start)
				startTIdx = tIdx;
		}
		printf(" (%d - %d)\n", max(startTIdx, 0), tIdx);

		// match / mismatch
		printf("   ");
		for (int j = start; j < start + 50 && j < alignmentLength; j++) {
			printf(alignment[j] == EDLIB_EDOP_MATCH ? "|" : " ");
		}
		printf("\n");

		// query
		printf("Q: ");
		int startQIdx = qIdx;
		for (int j = start; j < start + 50 && j < alignmentLength; j++) {
			if (alignment[j] == EDLIB_EDOP_DELETE)
				printf("-");
			else
				printf("%c", query[++qIdx]);
			if (j == start)
				startQIdx = qIdx;
		}
		printf(" (%d - %d)\n\n", max(startQIdx, 0), qIdx);
	}
}

