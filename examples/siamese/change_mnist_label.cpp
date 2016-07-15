//
// This script generate random mnist characters

#include <fstream>  // NOLINT(readability/streams)
#include <stdlib.h>
#include <stdio.h>

int main()
{
  std::ifstream infile ("./data/mnist/train-labels-idx1-ubyte",std::ifstream::binary);
  //std::ifstream labelfile ("./data/mnist/t10k-labels-idx1-ubyte",std::ifstream::binary);
  std::ofstream outfile ("./data/mnist/train-labels-idx1-ubyte-0c1",std::ofstream::binary);

  // get size of file
  infile.seekg (0,infile.end);
  long size = infile.tellg();
  infile.seekg (0);

  // allocate memory for file content
  char* buffer = new char[size];

  // read content of infile
  infile.read (buffer,size);
  int n0=0;
  int nChanged = 0;
  for(int i=0; i<60000; i++) {

    if(buffer[8+i]==0) {
      n0++;
      if (rand()%100 > 50) {
         buffer[8+i]=1;
         nChanged++;
      }
    }
  }

  // write to outfile
  outfile.write (buffer,size);

  // release dynamically-allocated memory
  delete[] buffer;

  outfile.close();
  infile.close();
  printf("There are %d 0s, and %d of them are labeled as 1\n", n0, nChanged);
  return 0;
}
