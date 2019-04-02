smsMap: main.o common.o edlib.o
	g++ -std=c++11 -fopenmp -g -o smsMap main.o common.o edlib.o
main.o: main.cpp common.h edlib.h
	g++ -std=c++11 -fopenmp -g -c main.cpp
common.o: common.cpp common.h edlib.h
	g++ -std=c++11 -fopenmp -g -c common.cpp
edlib.o: edlib.cpp edlib.h
	g++ -std=c++11 -fopenmp -g -c edlib.cpp
clean:
	rm *.o smsMap
