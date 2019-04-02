#include "common.h"
#pragma warning(disable:4996)
using namespace std;
Options options;
SequenceDB seq_db;
GenomeDB genome_db;

int main(int argc, const char *argv[])
{
	//int argc = 8;
	//if (argc < 2) {
	//	print_usage();
	//	return 0;
	//}
	if (argc < 8) {
                        print_usage_sequence_banded_alignment();
                        return 0;
        }

	string db_in;
	string genome_in;
	string db_out;
	string command;
	//clock_t start, finish;
	//start = clock();
	//double totaltime;
	float identify;
	//int argc = 8;
	int max_length = 0, max_banded_width = 0, kmerLength;
	//int **matrix; // score matrix
	//int **track; // track score
	//const char *argv[] = { "./smsAlign", "-locate", "-i","D:\\OTU拼接\\回国后文章\\数据\\ecoli真实数据\\ecoliCLRreads.fa", "-j" ,"D:\\OTU拼接\\回国后文章\\数据\\ecoli真实数据\\ecoli_polished_assembly4681865.fa", "-o" ,"ecoili_raw_pos.txt" };
	//const char *argv[] = { "./smsAlign","-locate","-i","D:\\OTU拼接\\回国后文章\\数据\\ecoli真实数据\\ecoliCLRreads.fa", "-lib", "ecoili_raw_lib.txt","-o" ,"ecoili_raw_pos.txt"}; 
	//const char *argv[] = { "./smsAlignPositionLocate","-i","D:\\OTU拼接\\回国后文章\\数据\\E.coli_genome_NCBI.4641652.fa.npbss_simulated_CLR50X.fa", "-j","D:\\OTU拼接\\回国后文章\\数据\\E.coli_genome_NCBI.4641652.Normalize.fasta", "-T","5","-lib" ,"genomeLib.txt", "-o", "sequencePosition.txt" };
	//const char *argv[] = { "./smsAlign","-align","-i","D:\\OTU拼接\\回国后文章\\数据\\模拟数据\\E.coli_genome_NCBI.4641652.fa.npbss_simulated_CLR50X.fa", "-j","D:\\OTU拼接\\回国后文章\\数据\\模拟数据\\E.coli_genome_NCBI.4641652.Normalize.fasta", "-o", "aligned.txt","-pos","seqPos.txt"};
	command = argv[1];
	//if (command.find("-align") != string::npos) {
	if (argc < 8) {
		print_usage_sequence_banded_alignment();
		return 0;
	}
	bool flag = options.SetOptions(argc, argv);
	options.Validate();
	if (flag) {
		db_in = options.input;   //options member
		genome_in = options.genome;
		fprintf(stdout, "Reading sequence file...");
		seq_db.Read(db_in.c_str(), options); // read the sequence file;
		fprintf(stdout, "    Done.\n\n");
		seq_db.SequenceStatistic(options);
		fprintf(stdout, "Reading sequence position file...");
		seq_db.ReadSequencePositionFile(options);
		fprintf(stdout, "    Done.\n\n");
		fprintf(stdout, "Reading genome file...");
		genome_db.Read_my(genome_in.c_str(), options); // read the genome file;
		genome_db.GenomeMinus();
		//printf("    Done.\n\n");
		fprintf(stdout, "Aligning sequences...");
		genome_db.seq_align_multi_threads(seq_db, options);
		//printf("    Done.\n\n");
		printf("\nWriting aligned sequences...");
		seq_db.WriteAlignedToFile(options);
		printf("    Done.\n\n");
	}
	//}
	//else {
	//	print_usage_sequence_banded_alignment();//print_usage();
	//	return 0;
	//}
	printf("Finished at: %s\n\n", getTime().c_str());
	//system("pause");
	return 0;
}
