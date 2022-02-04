/* maxtree.c */
/* October 2000  Erik Urbach */
#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "maxtree.h"
#include "mtmacros.h"
#include "gui.h"
#include "hdf5.h"

#define DATASET         "image"

typedef struct FIFOQueue
{
   long *Pixels;
   long Head;
   long Tail; /* First free place in queue, or -1 when the queue is full */
} FIFOQueue;




MaxTree Tree;



/*** Globals used by Max-Tree ***/

long numbernodes[NUMLEVELS]; /* Number of nodes C^k_h at level h */
double *ORI;  /* Denotes the original gray level value of pixel p */
short *RANK; /* denotes the rank of each pixel */
short *INDEX; /* denotes the index of each pixel to look up grey level in ORI */
double *SHAPE;

/* added by Jann */
double *ORIUnique;
int ORIUniqueSize;

/* STATUS stores the information of the pixel status: the pixel can be
 * NotAnalyzed, InTheQueue or assigned to node k at level h. In this
 * last case STATUS(p)=k. */
#define ST_NotAnalyzed  -1
#define ST_InTheQueue   -2
long *STATUS;

bool NodeAtLevel[NUMLEVELS];

/*** HQueue globals ***/
long MaxPixelsPerLevel[NUMLEVELS];
FIFOQueue HQueue[NUMLEVELS];

/*** Other globals ***/
long ImageWidth;
long ImageHeight;
long ImageSize;

long NumPixelsBelowLevel[NUMLEVELS];



long GetNumberOfNodes(int level)
{
   return(numbernodes[level]);
} /* GetNumberNodes */



long GetNumPixelsBelowLevel(int level)
{
   return(NumPixelsBelowLevel[level]);
} /* GetNumPixelsBelowLevel */



long GetNodeIndex(long p, int h)
{
   return(NumPixelsBelowLevel[h]+STATUS[p]);
} /* GetNodeIndex */



/*** Queue functions ***/
#define HQueueFirst(h)     (HQueue[h].Pixels[HQueue[h].Head++])
#define HQueueAdd(h,p)     HQueue[h].Pixels[HQueue[h].Tail++] = p
#define HQueueNotEmpty(h)  (HQueue[h].Head != HQueue[h].Tail)
void HQueueCreate(void);

void HQueueCreate(void)
{
   int i;

   HQueue->Pixels = calloc((size_t)ImageSize, sizeof(long));
   assert(HQueue->Pixels != NULL);
   HQueue->Head = HQueue->Tail = 0;
   for (i=1; i<NUMLEVELS; i++)
   {
      HQueue[i].Pixels = HQueue[i-1].Pixels + MaxPixelsPerLevel[i-1];
      HQueue[i].Head = HQueue[i].Tail = 0;
   }
} /* HQueueCreate */



#define ReadPixel(t,i,h)  (t[NumPixelsBelowLevel[h]+i].Level)



ubyte *ReadPGMAscii(char *fname, long *width, long *height)
{
   FILE *infile;
   ubyte *img;
   size_t size, i;
   int c;

   infile = fopen(fname, "r");
   if (infile==NULL)
   {
      fprintf(stderr, "Could not open file '%s'!\n", fname);
      return(NULL);
   }
   fscanf(infile, "P2\n");
   while ((c=fgetc(infile)) == '#')
      while ((c=fgetc(infile)) != '\n');
   ungetc(c, infile);
   fscanf(infile, "%ld %ld\n255\n", width, height);
   size = (size_t)((*width) * (*height));
   img = calloc(size, sizeof(ubyte));
   for (i=0; i<size; i++)
   {
      fscanf(infile, "%d", &c);
      img[i] = c;
   }
   fclose(infile);
   return(img);
} /* ReadPGMAscii */

ubyte *ReadPGMBin(char *fname, long *width, long *height)
{
   FILE *infile;
   ubyte *img;
   size_t size;
   int c;

   infile = fopen(fname, "rb");
   if (infile==NULL)
   {
      fprintf(stderr, "Could not open file '%s'!\n", fname);
      return(NULL);
   }
   fscanf(infile, "P5\n");
   while ((c=fgetc(infile)) == '#')
      while ((c=fgetc(infile)) != '\n');
   ungetc(c, infile);
   fscanf(infile, "%ld %ld\n255\n", width, height);
   size = (size_t)((*width) * (*height));
   img = calloc(size, sizeof(ubyte));
   if (img)  fread(img, 1, size, infile);
   fclose(infile);
   return(img);
} /* ReadPGMBin */

ubyte *ReadPGM(char *fname, long *width, long *height)
{
   FILE *infile;
   char id[4];

   infile = fopen(fname, "r");
   if (infile==NULL)
   {
      fprintf(stderr, "Could not open file '%s'!\n", fname);
      return(NULL);
   }
   fscanf(infile, "%3s", id);
   fclose(infile);
   if (strcmp(id, "P2")==0)  return(ReadPGMAscii(fname, width, height));
   else if (strcmp(id, "P5")==0)  return(ReadPGMBin(fname, width, height));
   else
   {
      fprintf(stderr, "Unknown image format\n");
      return(NULL);
   }
} /* ReadPGM */



void WritePGM(char *fname, long width, long height)
{
  FILE *outfile;

  outfile = fopen(fname, "wb");
  if (outfile==NULL)
  {
     fprintf(stderr, "Can't write file '%s'\n", fname);
     return;
  }
  fprintf(outfile, "P5\n%ld %ld\n255\n", width, height);
  fwrite(ORI, 1, (size_t)(width*height), outfile);
  fclose(outfile);
} /* WritePGM */

/* added by Jann */
double *ReadHDF(char *filename, long *width, long *height)
{
    hid_t       file, space, dset;          /* Handles */
    herr_t      status;
    hsize_t     dims[2] = {0};
    int         ndims,
                i, n, m,
                counter = 0,
                image_size = 48; /* width & height of the CTA input images */
    double      **rdata;                    /* Read buffer */

    /*
     * Open file and dataset.
     */
    file = H5Fopen (filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    dset = H5Dopen (file, DATASET, H5P_DEFAULT);
    space = H5Dget_space (dset);
    ndims = H5Sget_simple_extent_dims (space, dims, NULL);

    /*
     * Allocate array of pointers to rows.
     */
    rdata = (double **) malloc (dims[0] * sizeof (double *));

    /*
     * Allocate space.
     */
    rdata[0] = (double *) malloc (dims[0] * dims[1] * sizeof (double));

    /*
     * Set the rest of the pointers to rows to the correct addresses.
     */
    for (i=1; i<dims[0]; i++)
        rdata[i] = rdata[0] + i * dims[1];

    /*
     * Read the data.
     */
    status = H5Dread (dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                rdata[0]);
    /*
     * Output the data to the screen.
     */
    printf ("Input image: \n");
    for (n = 0; n < image_size; n++)
    {
        printf (" [");
        for (m = 0; m < image_size; m++)
        {
            printf (" %.1f", rdata[0][counter]);
            counter += 1;
        }
        printf ("]\n");
    }

    /*
     * Close, release resources and return image.
     */
    status = H5Dclose (dset);
    status = H5Sclose (space);
    status = H5Fclose (file);
    ImageWidth = image_size;
    ImageHeight = image_size;
    return(rdata[0]);
}

/* added by Jann */
double FindMaxPixelValue(double *array, long n)
{
    int i;
    double max = array[0];
 
    for (i = 1; i < n; i++)
        if (array[i] > max)
            max = array[i];

    return(max);
}

/* added by Jann */
double FindMinPixelValue(double *array, long n)
{
    int i;
    double min = array[0];
 
    for (i = 1; i < n; i++)
        if (array[i] < min)
            min = array[i];

    return(min);
}

/* added by Jann */
void Swap(double *a, double *b) 
{ 
double temp = *a; 
*a = *b; 
*b = temp; 
}  

/* added by Jann */
void BubbleSort(double *arr, int arrsize) 
{ 
int i, j; 
for (i = 0; i < arrsize - 1; i++) 
{
    for (j = 0; j < arrsize - i - 1; j++) 
    {
        if (arr[j] > arr[j+1])
        {
            Swap(&arr[j], &arr[j+1]); 
        }
    }
}      
}   

/* added by Jann */
double *UniqueElements(double *arr, int arrsize, int *sizeoutput)
{
    /* declare local variables */
    int i, j, k;
        
    /* use nested for loop to find the duplicate elements in array */
    for ( i = 0; i < arrsize; i ++)  
    {  
        for ( j = i + 1; j < arrsize; j++)  
        {  
            /* use if statement to check duplicate element */  
            if ( arr[i] == arr[j])  
            {  
                /* delete the current position of the duplicate element */  
                for ( k = j; k < arrsize - 1; k++)  
                {  
                    arr[k] = arr [k + 1];  
                }  
                /* decrease the size of array after removing duplicate element */
                arrsize--;  
                    
            /* if the position of the elements is changed, don't increase the index j */ 
                j--;      
            }  
        }  
    }

    /* sort ORI elements */
    BubbleSort(arr, arrsize);

    /* print sorted unique ORI elements */
    printf ("Unique ORI elements: [");  
    for ( i = 0; i < arrsize; i++)  
    {  
        printf (" %.1f ", arr[i]);  
    } 
    printf("]\n");

    /* print unique ORI elements size */
    printf("Number of unique ORI elements: %d \n", arrsize);

    *sizeoutput = arrsize;
    return(arr);  
}


/* Comparison function. Receives two generic (void) pointers to the items under comparison. */
int compare_indices(const void *p, const void *q) {
    short i = *(const int *)p;
    short j = *(const int *)q;

 
    if (i == j) return 0;
    
    if ( (ORI[i] < ORI[j] ) || ( (ORI[i]== ORI[j]) && ( i < j)) )
        return -1;
    
    return 1;   


}

/* Sort an array of n integers, pointed to by a. */
void sort_indices(short *a, size_t n) {
    qsort(a, n, sizeof(*a), compare_indices);
}

short *CreateIndex(int imagesize){
  short *indices = calloc(imagesize,sizeof(short));
  int i;
 
  for (i = 0; i<imagesize; i++){
    indices[i] = i;
  }
  
  sort_indices(indices, (size_t) imagesize);

  return indices;

}

short *CreateRanks( short *indices, int imagesize){
  short *ranks = calloc(imagesize,sizeof(short));
  int i;
  
  for (i = 0; i<imagesize; i++){
    ranks[indices[i]] = i;
  }

  return ranks;
}

void ReadDiatom(char *diatomname, char *shapename, double **diatomimage, double **shapeimage, long *width, long *height)
{
   long swidth, sheight, p;

   /* START adjusted/added by Jann */
   ORI = ReadHDF(diatomname, &ImageWidth, &ImageHeight);
   if (ORI==NULL)  exit(-1);
   ImageSize = ImageWidth*ImageHeight;
   printf("Min ORI value: %.1f \n", FindMinPixelValue(ORI, ImageSize));
   printf("Max ORI value: %.1f \n", FindMaxPixelValue(ORI, ImageSize));
   /*  ORIUnique = UniqueElements(ORI, ImageSize, &ORIUniqueSize);*/
   INDEX = CreateIndex(ImageSize);
   if (INDEX == NULL)
      {
         free(ORI);
         exit(-1);
      }
   RANK = CreateRanks(INDEX, ImageSize);

   if (RANK == NULL)
      {
        free(INDEX);
	free(ORI);
         exit(-1);
      }
   /* END adjusted/added by Jann */
   
   if (shapename)
   {
      SHAPE = ReadHDF(shapename, &swidth, &sheight);
      if (SHAPE==NULL)
      {
         free(ORI);
	 free(INDEX);
	 free(RANK);
         exit(-1);
      }
      if ((swidth!=ImageWidth) || (sheight!=ImageHeight))
      {
         fprintf(stderr, "Images have different sizes!\n");
         free(SHAPE);
  	 free(RANK);
	 free(INDEX);
	 free(ORI);
         exit(-1);
      }
   } else {
      SHAPE = calloc((size_t)ImageSize, sizeof(double));
      if (SHAPE==NULL)
      {
         free(ORI);
         free(INDEX);
 	 free(RANK);
         exit(-1);
      }
      for (p=0; p<ImageSize; p++)  SHAPE[p] = 255; 
   }
   *diatomimage = ORI;
   *shapeimage = SHAPE;
   *width = ImageWidth;
   *height = ImageHeight;
} /* ReadDiatom */



MaxTree MaxTreeCreate(long size)
/* PRE: MaxPixelsPerLevel initialized */
{
   MaxTree t;
   int i;

   t = calloc((size_t)size, sizeof(MaxNode));
   if (t)
   {
      *NumPixelsBelowLevel = 0;
      for (i=1; i<NUMLEVELS; i++)
      {
         NumPixelsBelowLevel[i] = NumPixelsBelowLevel[i-1] + MaxPixelsPerLevel[i-1];
      }
   }
   return(t);
} /* MaxTreeCreate */



long FindHMinPixel(void)
/* Finds a pixel of minimum RANK in the image */
{
   long i, p = -1;

   for (i=0; i<NUMLEVELS; i++)  MaxPixelsPerLevel[i] = 0;
   for (i=0; i<ImageSize; i++)
   {
      if (SHAPE[i])
      {
         MaxPixelsPerLevel[(int) RANK[i]]++;
         if (p>=0)
         {
            if (RANK[i]<RANK[p])  p = i;
         } else  p = i;
      }
   }
   return(p);
} /* FindHMinPixel */




int GetNeighbors(long p, long *neighbors)
{
   long x;
   int n=0;

   x = p % ImageWidth;
   if ((x<(ImageWidth-1)) && (SHAPE[p+1]))        neighbors[n++] = p+1;
   if ((p>=ImageWidth) && (SHAPE[p-ImageWidth]))  neighbors[n++] = p-ImageWidth;
   if ((x>0) && (SHAPE[p-1]))                     neighbors[n++] = p-1;
   p += ImageWidth;
   if ((p<ImageSize) && (SHAPE[p]))               neighbors[n++] = p;
   return(n);
} /* GetNeighbors */



/*** Max-Tree creation ***/

int flood(int h,
          void *(*NewAuxData)(int, int),
          void (*AddToAuxData)(void *, int, int),
          void (*MergeAuxData)(void *, void *),
          long *thisarea,
	  int  *thispeaklevel,
          void **thisattr)
{
   long neighbors[CONNECTIVITY];
   long p, q, idx;
   long area = *thisarea, childarea;
   int peaklevel = *thispeaklevel, childpeaklevel;
   MaxNode *node;
   void *attr = NULL, *childattr;
   int numneighbors, i, m, x, y;

   while(HQueueNotEmpty(h))
   {
      area++;
      p = HQueueFirst(h);
      x = p%ImageWidth;
      y = p/ImageWidth;
      if (attr)  (*AddToAuxData)(attr, x, y);
      else
      {
         attr = (*NewAuxData)(x, y);
         if (*thisattr)  (*MergeAuxData)(attr, *thisattr);
      }
      STATUS[p] = numbernodes[h];
      numneighbors = GetNeighbors(p, neighbors);
      for (i=0; i<numneighbors; i++)
      {
         q = neighbors[i];
         if (STATUS[q]==ST_NotAnalyzed)
         {
            HQueueAdd((int) RANK[q], q);
            STATUS[q] = ST_InTheQueue;
            NodeAtLevel[(int) RANK[q]] = true;
            if (RANK[q] > RANK[p])
            {
               m = RANK[q];
               childarea = 0;
               childpeaklevel = m;
               childattr = NULL;
               do
               {
		 m = flood(m, NewAuxData, AddToAuxData, MergeAuxData, &childarea, &childpeaklevel, &childattr);
               } while (m!=h);
               area += childarea;               
               (*MergeAuxData)(attr, childattr);
               if (childpeaklevel > peaklevel){
		 peaklevel = childpeaklevel;
	       }
            }
         }
      }
   }
   numbernodes[h] = numbernodes[h]+1;
   m = h-1;
   while ((m>=0) && (NodeAtLevel[m]==false))  m--;
   if (m>=0)
   {
      node = Tree + (NumPixelsBelowLevel[h] + numbernodes[h]-1);
      node->Parent = NumPixelsBelowLevel[m] + numbernodes[m];
   } else {
      idx = NumPixelsBelowLevel[h];
      node = Tree + idx;
      node->Parent = idx;
   }
   node->Area = area;
   node->PeakLevel = peaklevel;
   node->Attribute = attr;
   node->NodeStatus = MTS_Ok;
   node->Level = ORI[INDEX[h]];
   NodeAtLevel[h] = false;
   *thisarea = area;
   *thispeaklevel = peaklevel;
   *thisattr = attr;
   return(m);
} /* flood */

int floodMD(int h, int numattrs, ProcSet *procs, long *thisarea, int  *thispeaklevel, void **thisattrs)
{
   /* PRE: h - minimum intensity value
    *      numattrs - number of attributes to compute
    *      procs - array [0..numattrs-1] of attribute functions
    *      thisarea - (pointer to) number of already processed pixels connected with this conn. component.
    *      thisattrs - array [0..numattrs-1] of NULL pointers.
    * POST: procs - unchanged.
    *       thisarea - (pointer to) this conn. component's number of pixels.
    *       thisattrs - array filled with this conn. component's attribute values.
    */
   long neighbors[CONNECTIVITY];
   void **childattrs, **attrs = NULL;
   MaxNode *node;
   long area = *thisarea;
   int peaklevel = *thispeaklevel, childpeaklevel;
   long p, q, childarea, idx;
   int x, y, i, numneighbors, m, j;

   childattrs = calloc(numattrs, sizeof(void *));
   while(HQueueNotEmpty(h))
   {
      area++;
      p = HQueueFirst(h);
      x = p%ImageWidth;
      y = p/ImageWidth;
      if (attrs)
      {
         for (i=0; i<numattrs; i++)  procs[i].AddToAuxData(attrs[i], x, y);
      } else {
         attrs = calloc(numattrs, sizeof(void *));
         for (i=0; i<numattrs; i++)
         {
            attrs[i] = procs[i].NewAuxData(x, y);
            if (thisattrs[i])  procs[i].MergeAuxData(attrs[i], thisattrs[i]);
         }
      }
      STATUS[p] = numbernodes[h];
      numneighbors = GetNeighbors(p, neighbors);
      for (i=0; i<numneighbors; i++)
      {
         q = neighbors[i];
         if (STATUS[q]==ST_NotAnalyzed)
         {
            HQueueAdd((int) RANK[q], q);
            STATUS[q] = ST_InTheQueue;
            NodeAtLevel[(int) RANK[q]] = true;
            if (RANK[q] > RANK[p])
            {
               m = RANK[q];
               childarea = 0;
               childpeaklevel = m;
	       for (j=0; j<numattrs; j++)  childattrs[j] = NULL;
               do
               {
		 m = floodMD(m, numattrs, procs, &childarea, 
			     &childpeaklevel, childattrs);
               } while (m!=h);
               area += childarea;
               for (j=0; j<numattrs; j++){
		 procs[j].MergeAuxData(attrs[j], childattrs[j]);
	       }
               if (childpeaklevel > peaklevel){
		 peaklevel = childpeaklevel;
	       }

            }

         }
      }
   }
   numbernodes[h] = numbernodes[h]+1;
   m = h-1;
   while ((m>=0) && (NodeAtLevel[m]==false))  m--;
   if (m>=0)
   {
      node = Tree + (NumPixelsBelowLevel[h] + numbernodes[h]-1);
      node->Parent = NumPixelsBelowLevel[m] + numbernodes[m];
   } else {
      idx = NumPixelsBelowLevel[h];
      node = Tree + idx;
      node->Parent = idx;
   }
   node->Area = area;
   node->PeakLevel = peaklevel;
   node->Attribute = attrs;
   node->Pos = calloc(numattrs, sizeof(int));
   node->NodeStatus = MTS_Ok;
   node->Level = ORI[INDEX[h]];
   NodeAtLevel[h] = false;
   *thispeaklevel = peaklevel;
   *thisarea = area;
   for (i=0; i<numattrs; i++)  thisattrs[i] = attrs[i];
   return(m);
} /* floodMD */



void MaxTreeFilterMin(MaxTree t,
                      double (*Attribute)(void *),
                      double lambda)
{
   long i, idx, parent;
   int l;
   long numnodes=0, numdeleted=0;
   double sum=0.0;

   for (l=0; l<NUMLEVELS; l++)
   {
      numnodes += numbernodes[l];
      for (i=0; i<numbernodes[l]; i++)
      {
         idx = NumPixelsBelowLevel[l] + i;
         parent = t[idx].Parent;
         if ((idx!=parent) && ((t[parent].NodeStatus==MTS_Deleted)||((*Attribute)(t[idx].Attribute)<lambda)))
         {
            t[idx].NodeStatus = MTS_Deleted;
            t[idx].Level = t[parent].Level;
            numdeleted++;
         }
      }
   }
   for (i=0; i<ImageSize; i++)
   {
      if (SHAPE[i])
      {
         sum += ORI[i] - ReadPixel(t,STATUS[i], RANK[i]);
         ORI[i] = ReadPixel(t, STATUS[i], RANK[i]);
      }
   }
} /* MaxTreeFilterMin */

void MaxTreeFilterMDMin(MaxTree t, int numattrs, ProcSet *procs, double *lambdas)
{
   void **attrs;
   long i, idx, parent;
   int l, a;
   long numnodes=0, numdeleted=0;
   double sum=0.0;

   for (l=0; l<NUMLEVELS; l++)
   {
      numnodes += numbernodes[l];
      for (i=0; i<numbernodes[l]; i++)
      {
         idx = NumPixelsBelowLevel[l] + i;
         parent = t[idx].Parent;
         if (idx!=parent)
         {
            if (t[parent].NodeStatus!=MTS_Deleted)
            {
               attrs = t[idx].Attribute;
               for (a=0; a<numattrs; a++)
               {
                  if (procs[a].Attribute(attrs[a]) < lambdas[a])  break;
               }
               if (a<numattrs)
               {
                  t[idx].NodeStatus = MTS_Deleted;
                  t[idx].Level = t[parent].Level;
                  numdeleted++;
               }
            } else {
               t[idx].NodeStatus = MTS_Deleted;
               t[idx].Level = t[parent].Level;
               numdeleted++;
	    }
	 }
      }
   }
   for (i=0; i<ImageSize; i++)
   {
      if (SHAPE[i])
      {
         sum += ORI[i] - ReadPixel(t,STATUS[i], RANK[i]);
         ORI[i] = ReadPixel(t, STATUS[i], RANK[i]);
      }
   }
} /* MaxTreeFilterMDMin */

void MaxTreeGranulometryMin(MaxTree t,
                            double (*Attribute)(void *),
                            int (*LambdaMap)(double lambda, int num, double dlow, double dhigh),
                            int num,    /* number of result entries */
                            double dlow,
                            double dhigh,
                            double *result)
{
   long i, idx, parent;
   int l, lm, lam;

   for (l=0; l<=num; l++)  result[l] = 0.0;
   for (l=0; l<NUMLEVELS; l++)
   {
      for (i=0; i<numbernodes[l]; i++)
      {
         idx = NumPixelsBelowLevel[l] + i;
         parent = t[idx].Parent;
         if (idx!=parent)
         {
            lam = (*LambdaMap)((*Attribute)(t[idx].Attribute),num,dlow,dhigh);
            lm = MIN(t[parent].NodeStatus, lam);
            result[lm] += t[idx].Area*(t[idx].Level-t[parent].Level);
            t[idx].NodeStatus = lm;
         } else  t[idx].NodeStatus = num;
      }
   }
   for (l=0; l<NUMLEVELS; l++)
   {
      for (i=0; i<numbernodes[l]; i++)
      {
         idx = NumPixelsBelowLevel[l] + i;
         t[idx].NodeStatus = MTS_Ok;
      }
   }
} /* MaxTreeGranulometryMin */

void MaxTreeGranulometryMDMin(MaxTree t, int numattrs, ProcSet *procs, int *dims, double *result,
                              double *dls, double *dhs)
{
   void **attrs;
   long size, i, idx, parent, pos;
   int l, j, lm;

   size = dims[0]+1;
   for (l=1; l<numattrs; l++)  size *= dims[l]+1;
   for (i=0; i<size; i++)  result[i] = 0.0;
   for (l=0; l<NUMLEVELS; l++)
   {
      for (i=0; i<numbernodes[l]; i++)
      {
         idx = NumPixelsBelowLevel[l] + i;
         parent = t[idx].Parent;
         attrs = t[idx].Attribute;
         pos = 0;
	 for (j=numattrs-1; j>=0; j--)
	 {
            lm = procs[j].Mapper(procs[j].Attribute(attrs[j]), dims[j], dls[j], dhs[j]);
            if (idx!=parent)
            {
               if (parent!=t[parent].Parent)  t[idx].Pos[j] = MIN(lm, t[parent].Pos[j]);
               else  t[idx].Pos[j] = lm;
               pos = pos*(dims[j]+1) + t[idx].Pos[j];
            }
	 }
         if (idx!=parent)  result[pos] += t[idx].Area*(t[idx].Level-t[parent].Level);
      }
   }
} /* MaxTreeGranulometryMDMin */

void MaxTreeFilterDirect(MaxTree t,
                         double (*Attribute)(void *),
                         double lambda)
{
   long i, idx, parent;
   int l;
   long numnodes=0, numdeleted=0;
   double sum=0.0;

   for (l=0; l<NUMLEVELS; l++)
   {
      numnodes += numbernodes[l];
      for (i=0; i<numbernodes[l]; i++)
      {
         idx = NumPixelsBelowLevel[l] + i;
         parent = t[idx].Parent;
         if ((idx!=parent) && ((*Attribute)(t[idx].Attribute)<lambda))
         {
            t[idx].NodeStatus = MTS_Deleted;
            t[idx].Level = t[parent].Level;
            numdeleted++;
         }
      }
   }
   for (i=0; i<ImageSize; i++)
   {
      if (SHAPE[i])
      {
         sum += ORI[i] - ReadPixel(t,STATUS[i], RANK[i]);
         ORI[i] = ReadPixel(t, STATUS[i], RANK[i]);
      }
   }
} /* MaxTreeFilterDirect */

void MaxTreeFilterMDDirect(MaxTree t, int numattrs, ProcSet *procs, double *lambdas)
{
   void **attrs;
   long i, idx, parent;
   int l, a;
   long numnodes=0, numdeleted=0;
   double sum=0.0;

   for (l=0; l<NUMLEVELS; l++)
   {
      numnodes += numbernodes[l];
      for (i=0; i<numbernodes[l]; i++)
      {
         idx = NumPixelsBelowLevel[l] + i;
         parent = t[idx].Parent;
         if (idx!=parent)
         {
            attrs = t[idx].Attribute;
            for (a=0; a<numattrs; a++)
            {
               if (procs[a].Attribute(attrs[a]) < lambdas[a])  break;
            }
            if (a<numattrs)
            {
               t[idx].NodeStatus = MTS_Deleted;
               t[idx].Level = t[parent].Level;
               numdeleted++;
            }
         }
      }
   }
   for (i=0; i<ImageSize; i++)
   {
      if (SHAPE[i])
      {
         sum += ORI[i] - ReadPixel(t,STATUS[i], RANK[i]);
         ORI[i] = ReadPixel(t, STATUS[i], RANK[i]);
      }
   }
} /* MaxTreeFilterMDDirect */

void MaxTreeGranulometryDirect(MaxTree t,
                               double (*Attribute)(void *),
                               int (*LambdaMap)(double lambda, int num, double dlow, double dhigh),
                               int num,    /* number of result entries */
                               double dlow,
                               double dhigh,
                               double *result)
{
   long i, idx, parent, lm, lmin, cur;
   int l;

   for (l=0; l<=num; l++)  result[l] = 0;
   for (l=0; l<NUMLEVELS; l++)
   {
      for (i=0; i<numbernodes[l]; i++)
      {
         idx = NumPixelsBelowLevel[l] + i;
         parent = t[idx].Parent;
         if (idx!=parent)
         {
            lm = (*LambdaMap)((*Attribute)(t[idx].Attribute), num, dlow, dhigh);
            result[lm] += t[idx].Area * (t[idx].Level - t[parent].Level);
            t[idx].NodeStatus = lm;
            lmin = 0;
            while (t[parent].NodeStatus < lm)
            {
               cur = parent;
               parent = t[cur].Parent;
               if (t[cur].NodeStatus >= lmin)
               {
                  result[t[cur].NodeStatus] -= t[idx].Area * (t[cur].Level - t[parent].Level);
                  result[lm] += t[idx].Area * (t[cur].Level - t[parent].Level);
                  lmin = t[cur].NodeStatus;
               }
            }
         } else  t[idx].NodeStatus = num;
      }
   }
   for (l=0; l<NUMLEVELS; l++)
   {
      for (i=0; i<numbernodes[l]; i++)
      {
         idx = NumPixelsBelowLevel[l] + i;
         t[idx].NodeStatus = MTS_Ok;
      }
   }
} /* MaxTreeGranulometryDirect */

void MaxTreeGranulometryMDDirect(MaxTree t, int numattrs, ProcSet *procs, int *dims, double *result,
                                 double *dls, double *dhs)
{
   void **attrs;
   double d;
   long i, size, idx, parent, pos, area, maxpos;
   int l, j, lm;

   size = dims[0]+1;
   for (j=1; j<numattrs; j++)  size *= dims[j]+1;
   for (i=0; i<size; i++)  result[i] = 0.0;
   for (l=0; l<NUMLEVELS; l++)
   {
      for (i=0; i<numbernodes[l]; i++)
      {
         idx = NumPixelsBelowLevel[l] + i;
         parent = t[idx].Parent;
         if (idx!=parent)
         {
            t[idx].NodeStatus = t[idx].Area;
            attrs = t[idx].Attribute;
	    pos = 0;
	    for (j=numattrs-1; j>=0; j--)
	    {
               pos = pos*(dims[j]+1) + procs[j].Mapper(procs[j].Attribute(attrs[j]), dims[j], dls[j], dhs[j]);
            }
            result[pos] += t[idx].Area*(t[idx].Level-t[parent].Level);
         }
      }
   }
   for (l=NUMLEVELS-1; l>=0; l--)
   {
      for (i=0; i<numbernodes[l]; i++)
      {
         idx = NumPixelsBelowLevel[l] + i;
         parent = t[idx].Parent;
	 area = t[idx].NodeStatus;
         attrs = t[idx].Attribute;
	 for (j=numattrs-1; j>=0; j--)
	 {
            t[parent].Pos[j] = procs[j].Mapper(procs[j].Attribute(attrs[j]), dims[j], dls[j], dhs[j]);
         }
	 idx = parent;
	 parent = t[parent].Parent;
	 while (idx!=parent)
         {
            attrs = t[idx].Attribute;
	    pos = 0;
	    maxpos = 0;
	    for (j=numattrs-1; j>=0; j--)
	    {
               lm = procs[j].Mapper(procs[j].Attribute(attrs[j]), dims[j], dls[j], dhs[j]);
	       t[parent].Pos[j] = MAX(lm, t[idx].Pos[j]);
               pos = pos*(dims[j]+1) + lm;
               maxpos = maxpos*(dims[j]+1) + t[parent].Pos[j];
            }
	    if (maxpos>pos)
	    {
               d = area*(t[idx].Level-t[parent].Level);
	       result[pos] -= d;
	       result[maxpos] += d;
	       t[idx].NodeStatus -= area;
               idx = parent;
               parent = t[parent].Parent;
	    } else  idx=parent;
         }
      }
   }
   for (l=0; l<NUMLEVELS; l++)
   {
      for (i=0; i<numbernodes[l]; i++)
      {
         idx = NumPixelsBelowLevel[l] + i;
	 t[idx].NodeStatus = 0;
      }
   }
} /* MaxTreeGranulometryMDDirect */

void MaxTreeFilterMax(MaxTree t,
                      double (*Attribute)(void *),
                      double lambda)
{
   long i, idx, parent;
   int l, h;
   long numnodes=0, numdeleted=0;
   double sum=0.0;

   for (l=NUMLEVELS-1; l>=0; l--)
   {
      numnodes += numbernodes[l];
      for (i=0; i<numbernodes[l]; i++)
      {
         idx = NumPixelsBelowLevel[l] + i;
         parent = t[idx].Parent;
         if (t[idx].NodeStatus==MTS_Processed)  t[parent].NodeStatus = MTS_Processed;
         else if (t[idx].NodeStatus==MTS_Ok)
         /* t[idx] is a leaf node */
	      {
            while((idx!=parent)&&(t[idx].NodeStatus==MTS_Ok)&&((*Attribute)(t[idx].Attribute)<lambda))
	         {
               t[idx].NodeStatus = MTS_Deleting;
	            idx = parent;
	            parent = t[parent].Parent;
	         }
            if ((idx!=parent) && (t[idx].NodeStatus==MTS_Ok))
	         {
	            while ((idx!=parent) && (t[idx].NodeStatus!=MTS_Processed))
	            {
                  t[idx].NodeStatus = MTS_Processed;
                  idx = parent;
                  parent = t[parent].Parent;
	            }
	         }
	      }
      }
   }
   for (l=NUMLEVELS-1; l>=0; l--)
   {
      for (i=0; i<numbernodes[l]; i++)
      {
         idx = NumPixelsBelowLevel[l] + i;
         parent = t[idx].Parent;
         while ((idx!=parent) && (t[idx].NodeStatus==MTS_Deleting))
	      {
	         idx = parent;
	         parent = t[parent].Parent;
	      }
         h = t[idx].Level;
         idx = NumPixelsBelowLevel[l] + i;
         parent = t[idx].Parent;
         while ((idx!=parent) && (t[idx].NodeStatus==MTS_Deleting))
	      {
            t[idx].NodeStatus = MTS_Deleted;
            t[idx].Level = h;
	         idx = parent;
	         parent = t[parent].Parent;
            numdeleted++;
	      }
      }
   }
   for (i=0; i<ImageSize; i++)
   {
      if (SHAPE[i])
      {
         sum += ORI[i] - ReadPixel(t,STATUS[i], RANK[i]);
         ORI[i] = ReadPixel(t, STATUS[i], RANK[i]);
      }
   }
} /* MaxTreeFilterMax */

void MaxTreeFilterMDMax(MaxTree t, int numattrs, ProcSet *procs, double *lambdas)
{
   void **attrs;
   long i, idx, parent;
   int l, h, a;
   long numnodes=0, numdeleted=0;
   double sum=0.0;
   bool deleting;

   for (l=NUMLEVELS-1; l>=0; l--)
   {
      numnodes += numbernodes[l];
      for (i=0; i<numbernodes[l]; i++)
      {
         idx = NumPixelsBelowLevel[l] + i;
         parent = t[idx].Parent;
         if (t[idx].NodeStatus==MTS_Processed)  t[parent].NodeStatus = MTS_Processed;
         else if (t[idx].NodeStatus==MTS_Ok)
         /* t[idx] is a leaf node */
	 {
            deleting = true;
            while ((idx!=parent) && (deleting))
            {
               attrs = t[idx].Attribute;
               for (a=0; a<numattrs; a++)
               {
                  if (procs[a].Attribute(attrs[a]) < lambdas[a])  break;
               }
               if (a<numattrs)
               {
                  t[idx].NodeStatus = MTS_Deleting;
                  idx = parent;
                  parent = t[parent].Parent;
	       } else  deleting = false;
            }
            if ((idx!=parent) && (t[idx].NodeStatus==MTS_Ok))
	    {
	       while ((idx!=parent) && (t[idx].NodeStatus!=MTS_Processed))
	       {
                  t[idx].NodeStatus = MTS_Processed;
                  idx = parent;
                  parent = t[parent].Parent;
	       }
	    }
         }
      }
   }
   for (l=NUMLEVELS-1; l>=0; l--)
   {
      for (i=0; i<numbernodes[l]; i++)
      {
         idx = NumPixelsBelowLevel[l] + i;
         parent = t[idx].Parent;
         while ((idx!=parent) && (t[idx].NodeStatus==MTS_Deleting))
	 {
	    idx = parent;
	    parent = t[parent].Parent;
	 }
         h = t[idx].Level;
         idx = NumPixelsBelowLevel[l] + i;
         parent = t[idx].Parent;
         while ((idx!=parent) && (t[idx].NodeStatus==MTS_Deleting))
	 {
            t[idx].NodeStatus = MTS_Deleted;
            t[idx].Level = h;
	    idx = parent;
	    parent = t[parent].Parent;
            numdeleted++;
	 }
      }
   }
   for (i=0; i<ImageSize; i++)
   {
      if (SHAPE[i])
      {
         sum += ORI[i] - ReadPixel(t,STATUS[i], RANK[i]);
         ORI[i] = ReadPixel(t, STATUS[i], RANK[i]);
      }
   }
} /* MaxTreeFilterMDMax */

void MaxTreeGranulometryMax(MaxTree t,
                            double (*Attribute)(void *),
                            int (*LambdaMap)(double lambda, int num, double dlow, double dhigh),
                            int num,    /* number of result entries */
                            double dlow,
                            double dhigh,
                            double *result)
{
   long i, idx, parent;
   int l, lm, lam;

   for (l=0; l<=num; l++)  result[l] = 0.0;
   for (l=NUMLEVELS-1; l>=0; l--)
   {
      for (i=0; i<numbernodes[l]; i++)
      {
         idx = NumPixelsBelowLevel[l] + i;
         parent = t[idx].Parent;
         if (idx!=parent)
         {
            lam = (*LambdaMap)((*Attribute)(t[idx].Attribute),num,dlow,dhigh);
            lm = MAX(t[idx].NodeStatus, lam);
            result[lm] += t[idx].Area*(t[idx].Level-t[parent].Level);
            t[parent].NodeStatus = MAX(lm, t[parent].NodeStatus);
         }
      }
   }
   for (l=0; l<NUMLEVELS; l++)
   {
      for (i=0; i<numbernodes[l]; i++)
      {
         idx = NumPixelsBelowLevel[l] + i;
         t[idx].NodeStatus = MTS_Ok;
      }
   }
} /* MaxTreeGranulometryMax */

void MaxTreeGranulometryMDMax(MaxTree t, int numattrs, ProcSet *procs, int *dims, double *result,
                              double *dls, double *dhs)
{
   void **attrs;
   long size, i, idx, parent, pos;
   int l, j, lm;

   size = dims[0]+1;
   for (l=1; l<numattrs; l++)  size *= dims[l]+1;
   for (i=0; i<size; i++)  result[i] = 0.0;
   for (l=NUMLEVELS-1; l>=0; l--)
   {
      for (i=0; i<numbernodes[l]; i++)
      {
         idx = NumPixelsBelowLevel[l] + i;
         parent = t[idx].Parent;
	 if (idx!=parent)
         {
            attrs = t[idx].Attribute;
            pos = 0;
            for (j=numattrs-1; j>=0; j--)
	    {
               lm = procs[j].Mapper(procs[j].Attribute(attrs[j]), dims[j], dls[j], dhs[j]);
	       if (t[idx].NodeStatus)  lm = MAX(lm, t[idx].Pos[j]);
	       if (t[parent].NodeStatus)  t[parent].Pos[j] = MAX(t[parent].Pos[j], lm);
               else  t[parent].Pos[j] = lm;
               pos *= dims[j] + 1;
               pos += lm;
	    }
            result[pos] += t[idx].Area*(t[idx].Level-t[parent].Level);
            t[idx].NodeStatus = 0;
            t[parent].NodeStatus = 1;
	 }
      }
   }
} /* MaxTreeGranulometryMDMax */

void MaxTreeFilterWilkinson(MaxTree t,
                            double (*Attribute)(void *),
                            double lambda)
{
   long i, idx, parent;
   int l;
   long numnodes=0, numdeleted=0;
   double sum=0.0;

   for (l=0; l<NUMLEVELS; l++)
   {
      numnodes += numbernodes[l];
      for (i=0; i<numbernodes[l]; i++)
      {
         idx = NumPixelsBelowLevel[l] + i;
         parent = t[idx].Parent;
         t[idx].NodeStatus = t[parent].NodeStatus;
         if ((idx!=parent) && ((*Attribute)(t[idx].Attribute)<lambda))
         {
            t[idx].NodeStatus += t[parent].Level - t[idx].Level;
            numdeleted++;
         }
      }
   }
   for (l=0; l<NUMLEVELS; l++)
   {
      for (i=0; i<numbernodes[l]; i++)
      {
         idx = NumPixelsBelowLevel[l] + i;
         parent = t[idx].Parent;
         if ((idx!=parent) && (t[idx].NodeStatus<0))  t[idx].Level += t[idx].NodeStatus;
      }
   }
   for (i=0; i<ImageSize; i++)
   {
      if (SHAPE[i])
      {
         sum += ORI[i] - ReadPixel(t,STATUS[i], RANK[i]);
         ORI[i] = ReadPixel(t, STATUS[i], RANK[i]);
      }
   }
} /* MaxTreeFilterWilkinson */

void MaxTreeFilterMDWilkinson(MaxTree t, int numattrs, ProcSet *procs, double *lambdas)
{
   void **attrs;
   long i, idx, parent;
   int l, a;
   long numnodes=0, numdeleted=0;
   double sum=0.0;

   for (l=0; l<NUMLEVELS; l++)
   {
      numnodes += numbernodes[l];
      for (i=0; i<numbernodes[l]; i++)
      {
         idx = NumPixelsBelowLevel[l] + i;
         parent = t[idx].Parent;
         t[idx].NodeStatus = t[parent].NodeStatus;
         if (idx!=parent)
         {
            attrs = t[idx].Attribute;
            for (a=0; a<numattrs; a++)
            {
               if (procs[a].Attribute(attrs[a]) < lambdas[a])  break;
            }
            if (a<numattrs)
            {
               t[idx].NodeStatus += t[parent].Level - t[idx].Level;
               numdeleted++;
            }
         }
      }
   }
   for (l=0; l<NUMLEVELS; l++)
   {
      for (i=0; i<numbernodes[l]; i++)
      {
         idx = NumPixelsBelowLevel[l] + i;
         parent = t[idx].Parent;
         if ((idx!=parent) && (t[idx].NodeStatus<0))  t[idx].Level += t[idx].NodeStatus;
      }
   }
   for (i=0; i<ImageSize; i++)
   {
      if (SHAPE[i])
      {
         sum += ORI[i] - ReadPixel(t,STATUS[i], RANK[i]);
         ORI[i] = ReadPixel(t, STATUS[i], RANK[i]);
      }
   }
} /* MaxTreeFilterMDWilkinson */

void MaxTreeGranulometryWilkinson(MaxTree t,
                                  double (*Attribute)(void *),
                                  int (*LambdaMap)(double lambda, int num, double dlow, double dhigh),
                                  int num,    /* number of result entries */
                                  double dlow,
                                  double dhigh,
                                  double *result)
{
   long i, idx, parent;
   int l;

   for (l=0; l<=num; l++)  result[l] = 0;
   for (l=0; l<NUMLEVELS; l++)
   {
      for (i=0; i<numbernodes[l]; i++)
      {
         idx = NumPixelsBelowLevel[l] + i;
         parent = t[idx].Parent;
         if (idx!=parent)
         {
            result[(*LambdaMap)((*Attribute)(t[idx].Attribute),num,dlow,dhigh)]+=t[idx].Area*(t[idx].Level-t[parent].Level);
         }
      }
   }
} /* MaxTreeGranulometryWilkinson */

void MaxTreeGranulometryMDWilkinson(MaxTree t, int numattrs, ProcSet *procs,
				    int *dims, double *result,
                                    double *dls, double *dhs)
{
   void **vec_attr;
   long i, size=0, idx, parent, pos;
   int l, j;

   
   size = dims[0]+1;

   for (j=1; j<numattrs; j++)  size *= dims[j]+1;

   for (i=0; i<size; i++){
     result[i] = 0.0;
   }
   
   for (l=0; l<NUMLEVELS; l++)
     {  
       for (i=0; i<numbernodes[l]; i++){
	  
         idx = NumPixelsBelowLevel[l] + i;
         parent = t[idx].Parent;
         if (idx!=parent)
	   {
	     vec_attr = t[idx].Attribute;
	     pos = 0;
	     for (j=numattrs-1; j>=0; j--)
	       { 
		 pos = pos*(dims[j]+1) + procs[j].Mapper(procs[j].Attribute(vec_attr[j]), dims[j], dls[j], dhs[j]);
		 
	       }
	     result[pos] += ((double)(t[idx].Area))*(t[idx].Level-t[parent].Level);
	    
	   }
       }
     }
}
 /* MaxTreeGranulometryMDWilkinson */
void MaxTreeGranulometryMDWilkinsonK(MaxTree t, int numattrs, ProcSet *procs, int *dims, double *result,
                                    double *dls, double *dhs, int k)
{
   void **vec_attr;
   long i, size, idx, parent, pos;
   int l, j;

   size = dims[0]+1;
   for (j=1; j<numattrs; j++)  size *= dims[j]+1;
   for (i=0; i<size; i++)  result[i] = 0.0;
   for (l=0; l<NUMLEVELS; l++)
   {
      for (i=0; i<numbernodes[l]; i++)
      {
         idx = NumPixelsBelowLevel[l] + i;
         parent = t[idx].Parent;
         if (idx!=parent)
         {
            vec_attr = t[idx].Attribute;
	    pos = 0;
            if ( ((t[idx].PeakLevel- t[idx].Level) * NUMLEVELS/4)>= (k * (NUMLEVELS-t[idx].Level) )){
	      for (j=numattrs-1; j>=0; j--) {
               pos = pos*(dims[j]+1) + 
		 procs[j].Mapper(procs[j].Attribute(vec_attr[j]), 
				 dims[j], dls[j], dhs[j]);
	      }
	      result[pos] += t[idx].Area*(t[idx].Level-t[parent].Level);
              t[idx].NodeStatus=MTS_ADDED;
	    }
         }
      }
   }
} /* MaxTreeGranulometryMDWilkinson */

void MaxTreeGranulometryMDWilkinsonK2(MaxTree t, int numattrs, ProcSet *procs, int *dims, double *result,
				     double *dls, double *dhs, int k)
{
   void **vec_attr;
   long i, size, idx, parent, pos, parpos, PAR;
   int l, j;

   size = dims[0]+1;
   for (j=1; j<numattrs; j++)  size *= dims[j]+1;
   for (i=0; i<size; i++)  result[i] = 0.0;
   for (l=NUMLEVELS-1; l>0; l--)
     {  /*just to make sure all is set to go */ 
      for (i=0; i<numbernodes[l]; i++)
      {
         idx = NumPixelsBelowLevel[l] + i;
         t[idx].NodeStatus=MTS_Ok;
      }
   }
   for (l=NUMLEVELS-1; l>0; l--)
   {
      for (i=0; i<numbernodes[l]; i++)
      {
         idx = NumPixelsBelowLevel[l] + i;
         if (t[idx].NodeStatus == MTS_Ok){
	   parent = t[idx].Parent;
	   if (idx!=parent)
	     {
	       vec_attr = t[idx].Attribute;
	       pos = 0;
	       for (j=numattrs-1; j>=0; j--)
		 {
		   pos = pos*(dims[j]+1) + 
		     procs[j].Mapper( procs[j].Attribute(vec_attr[j]), 
				      dims[j], dls[j], dhs[j]);
		 }
	       
	       vec_attr = t[parent].Attribute;
	       parpos = 0;
	       for (j=numattrs-1; j>=0; j--)
		 {
		   parpos = parpos*(dims[j]+1) + 
		     procs[j].Mapper( procs[j].Attribute(vec_attr[j]), 
				      dims[j], dls[j], dhs[j]);
		 }
	       
	       while (parpos==pos){
		 parent=t[parent].Parent;
		 if (parent!=t[parent].Parent){
		   vec_attr = t[parent].Attribute;
		   parpos = 0;
		   for (j=numattrs-1; j>=0; j--)
		     {
		       parpos = parpos*(dims[j]+1) + 
			 procs[j].Mapper( procs[j].Attribute(vec_attr[j]), 
					  dims[j], dls[j], dhs[j]);
		     }
		 } else { 
		   parpos = -1;  
		 }
	       }
	       PAR = parent;
	       if ((t[idx].Level-t[PAR].Level)>=k){
		 
		 parent = t[idx].Parent;
		 while ( (parent!=PAR) && (t[idx].NodeStatus==MTS_Ok) ){
		   result[pos] += t[idx].Area*(t[idx].Level-t[parent].Level);
                   t[idx].NodeStatus = MTS_ADDED;
                   idx = parent;
		   parent = t[idx].Parent;
		   
		 }
	       }
	     }
	 }
      }
   }
} /* MaxTreeGranulometryMDWilkinsonK */


void InitLevels(void)
{
   int i;

   for (i=0; i<NUMLEVELS; i++)
   {
      numbernodes[i] = 0;
      NodeAtLevel[i] = false;
   }
} /* InitLevels */



MaxTree MaxTreeBuild(void *(*NewAuxData)(int, int),
                     void (*AddToAuxData)(void *, int, int),
                     void (*MergeAuxData)(void *, void *))
{
   long i, p;
   long area = 0;
   void *attr = NULL;
   int hmin, peaklevel;

   STATUS = calloc((size_t)ImageSize, sizeof(long));
   assert(STATUS!=NULL);
   for (i=0; i<ImageSize; i++)  STATUS[i] = ST_NotAnalyzed;
   InitLevels();

   p = FindHMinPixel();
   hmin = RANK[p];
   peaklevel = hmin;
   NodeAtLevel[hmin] = true;
   HQueueCreate();
   HQueueAdd(hmin, p); STATUS[p] = ST_InTheQueue;
   Tree = MaxTreeCreate(ImageSize);
   assert(Tree!=NULL);
   flood(hmin, NewAuxData, AddToAuxData, MergeAuxData, &area, &peaklevel, &attr);
   return(Tree);
} /* MaxTreeBuild */



void MaxTreeDestroy(void (*DeleteAuxData)(void *))
{
   long i;
   int h;

   for (h=0; h<NUMLEVELS; h++)
   {
      for (i=0; i<numbernodes[h]; i++)
      {
         (*DeleteAuxData)(Tree[NumPixelsBelowLevel[h]+i].Attribute);
      }
   }
   free(Tree);
   free(HQueue->Pixels);
   free(STATUS);
} /* MaxTreeDestroy */



MaxTree MaxTreeBuildMD(int num, ProcSet *procs)
{
   long i, p;
   long area = 0;
   void **attrs;
   int hmin,peaklevel;

   STATUS = calloc((size_t)ImageSize, sizeof(long));
   if (STATUS==NULL)  return(NULL);
   for (i=0; i<ImageSize; i++)  STATUS[i] = ST_NotAnalyzed;
   InitLevels();

   p = FindHMinPixel();
   hmin = RANK[p];
   peaklevel = hmin;
   NodeAtLevel[hmin] = true;
   HQueueCreate();
   HQueueAdd(hmin, p); STATUS[p] = ST_InTheQueue;
   Tree = MaxTreeCreate(ImageSize);
   if (Tree==NULL)
   {
      free(HQueue->Pixels);
      free(STATUS);
      return(NULL);
   }
   attrs = calloc(num, sizeof(void *));
   if (attrs==NULL)
   {
      free(Tree);
      free(HQueue->Pixels);
      free(STATUS);
      return(NULL);
   }
   floodMD(hmin, num, procs, &area, &peaklevel, attrs);
   free(attrs);
   return(Tree);
} /* MaxTreeBuildMD */



void MaxTreeDestroyMD(int num, ProcSet *procs)
{
   void **vec_attr;
   MaxNode *node;
   long i, j;
   int h;

   for (h=0; h<NUMLEVELS; h++)
   {
      for (i=0; i<numbernodes[h]; i++)
      {
         node = &(Tree[NumPixelsBelowLevel[h]+i]);
         free(node->Pos);
         vec_attr = node->Attribute;
         for (j=0; j<num; j++)  procs[j].DeleteAuxData(vec_attr[j]);
         free(vec_attr);
      }
   }
   free(Tree);
   free(HQueue->Pixels);
   free(STATUS);
} /* MaxTreeDestroyMD */
