# include <stdio.h>
# include <math.h>
# include <ctype.h>
# include <stdlib.h>
# include <string.h>
# include <time.h>
#define MAXCHAR 128
#define MAXCHAR2 12800
#define MAXENTRY 200000
#define MAXBIT 5000
#define COLORTEXT "YES"

/* this program reads in a output of hex2ani and calculate Tanimoto similarity*/
typedef struct {
	char name[MAXCHAR];
	char bitstr[MAXBIT];
} FP;
FP ref;
FP input;
FP mcs;
FP maxref;
FILE *fpin;
FILE *fpout;
char ifilename[MAXCHAR];
char rfilename[MAXCHAR];
char mfilename[MAXCHAR];
char ofilename[MAXCHAR];
int i,j;
int pos[MAXBIT];
int nbit=0;
int nbit_r=0;
int nbit_i=0;
int nbit_m=0;
int nbit_m2 = 0;
int nref=0;
int i_mcs = 0;
int i_condition=0;
int nmatch;
double coef=0;
double maxcoef=0;
double cutoff = 0.85;

/*t(A,B) = A.B/(A*A + B*B - A.B)*/
double tanimoto_coef() {
int i, nr, ni, nc;
double a2, b2, ab;	
	nr = 0;
	ni = 0;	
	nc = 0;
	for(i=0;i<nbit_i;i++) {
		if(ref.bitstr[i] == '1') nr ++;
		if(input.bitstr[i] == '1') ni ++;
		if(ref.bitstr[i] == '1' && input.bitstr[i] == '1') nc++;
	}
	if(nc == 0 || nr == 0 || ni == 0) coef = 0;
	else {
		coef = 1.0 * nc/(nr + ni - nc);
	}
/*	printf("Tanimoto Coefficient is %10.6lf\n", coef); */
}

int main(int argc, char* argv[]){
int num;
int tmpint;
char line[MAXCHAR2];
if (strcmp(COLORTEXT, "YES") == 0 || strcmp(COLORTEXT, "yes") == 0) {
	if (argc == 2
		&& (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "-H") == 0)) {
		printf
			("[31mUsage: tanimoto_ss_mcs -r[0m reference FP\n"
			 "[31m                       -m[0m mcs FP\n"
			 "[31m                       -i[0m input FP\n"
			 "[31m                       -o[0m output file\n"
			 "[31m                       -c[0m cutoff, default is 0.85\n");
		exit(0);
	}
	if(argc != 5 && argc != 7 && argc != 9 && argc != 11) {
		printf
			("[31mUsage: tanimoto_ss_mcs -r[0m reference FP\n"
			 "[31m                       -m[0m mcs FP\n"
			 "[31m                       -i[0m input FP\n"
			 "[31m                       -o[0m output file\n"
			 "[31m                       -c[0m cutoff, default is 0.85\n");
		exit(0);
	}
}
else {
	if (argc == 2
		&& (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "-H") == 0)) {
		printf
			("Usage: tanimoto_ss_mcs -r reference FP\n"
			 "                       -m mcs FP\n"
			 "                       -i input FP\n"
			 "                       -o output file\n"
			 "                       -c cutoff, default is 0.85\n");
			exit(0);
		}
	if(argc != 5 && argc != 7 && argc != 9 && argc != 11) {
		printf
			("Usage: tanimoto_ss_mcs -r reference FP\n"
			 "                       -m mcs FP\n"
			 "                       -i input FP\n"
			 "                       -o output file\n"
			 "                       -c cutoff, default is 0.85\n");
			exit(0);
		}
}

for (i = 1; i < argc; i += 2) {
        if (strcmp(argv[i], "-r") == 0)
                strcpy(rfilename, argv[i + 1]);
        if (strcmp(argv[i], "-m") == 0) {
		i_mcs=1;
                strcpy(mfilename, argv[i + 1]);
	}
        if (strcmp(argv[i], "-i") == 0)
                strcpy(ifilename, argv[i + 1]);
        if (strcmp(argv[i], "-o") == 0)
                strcpy(ofilename, argv[i + 1]);
        if (strcmp(argv[i], "-c") == 0)
		cutoff=atof(argv[i+1]);
}

if ((fpin = fopen(ifilename, "r")) == NULL) {
        fprintf(stderr, "Cannot open input file %s, exit\n", ifilename);
        exit(0);
}
if ((fpout = fopen(ofilename, "w")) == NULL) {
        fprintf(stderr, "Cannot open output file %s, exit\n", ofilename);
        exit(0);
}
for (;;) {
        strcpy(line, "");
        line[0] = '\0';
        if (fgets(line, MAXCHAR2, fpin) == NULL) break;
        if(strlen(line) <= 1) continue;
        if(line[0]=='#') continue;
	sscanf(line, "%s%s", input.name,input.bitstr);
	break;
}
nbit_i = strlen(input.bitstr);
fclose(fpin);
if(i_mcs == 1) {
	if ((fpin = fopen(mfilename, "r")) == NULL) {
        	fprintf(stderr, "Cannot open mcs file %s, exit\n", mfilename);
        	exit(0);
	}
	for (;;) {
        	strcpy(line, "");
        	line[0] = '\0';
        	if (fgets(line, MAXCHAR2, fpin) == NULL) break;
        	if(strlen(line) <= 1) continue;
        	if(line[0]=='#') continue;
		sscanf(line, "%s%s", mcs.name,mcs.bitstr);
		break;
	}
	nbit_m = strlen(mcs.bitstr);
	if(nbit_i != nbit_m) {
		fprintf(stderr, "Different lengths of strings between input and mcs: %5d, %5d for %s\n", nbit_i, nbit_m, ref.name);
		exit(1);
	}
	fclose(fpin);
	nbit_m2 = 0;
	for(i=0; i< nbit_m; i++)
		if(mcs.bitstr[i] == '1') 
			nbit_m2++;
}
if ((fpin = fopen(rfilename, "r")) == NULL) {
        fprintf(stderr, "Cannot open reference file %s, exit\n", rfilename);
        exit(0);
}
for (;;) {
        strcpy(line, "");
        line[0] = '\0';
        if (fgets(line, MAXCHAR2, fpin) == NULL) break;
        if(strlen(line) <= 1) continue;
        if(line[0]=='#') continue;
        sscanf(line, "%s%s", ref.name,ref.bitstr);
	nbit_r = strlen(ref.bitstr);
	if(nbit_i != nbit_r) {
		fprintf(stderr, "Different lengths of strings: %5d, %5d for %s\n", nbit_i, nbit_r, ref.name);
		continue;
	}
	tanimoto_coef();
	if(coef > maxcoef) {
		maxcoef=coef;
		maxref=ref;
	}
	if(coef >= cutoff) {
		i_condition = 1;
		nmatch=-1;
		if(i_mcs == 1) {
			nmatch = 0;
			for(i=0;i<nbit_m;i++) 
				if(mcs.bitstr[i] == '1' && ref.bitstr[i] == '0') nmatch++;
		}
		fprintf(fpout, "%-10s %s %9.4lf %5d %8.2lf\n", ref.name, ref.bitstr, coef, nmatch, 1.0-1.0* nmatch/nbit_m2);
	}
}
if(i_condition == 0) {
	nmatch=-1;
	if(i_mcs == 1) {
		nmatch = 0;
		for(i=0;i<nbit_m;i++) 
			if(mcs.bitstr[i] == '1' && ref.bitstr[i] == '0') nmatch++;
	}
	fprintf(fpout, "%-10s %s %9.4lf %5d %8.2lf\n", maxref.name, maxref.bitstr, maxcoef, nmatch, 1.0-1.0*nmatch/nbit_m2);
}

fclose(fpin);
printf("\n");
return(1);
}
