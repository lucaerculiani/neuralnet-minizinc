include "increasing.mzn";
include "at_least.mzn";




function array[int] of var bool: bdot (array[int] of var bool: one, array[int] of var bool: other, set of int: len)=
    [one[i] /\ other[i] | i in len];

array[images, rows, columns] of int: int_images;
array[images] of bool: labels;


int: ROWS;
int: COLUMNS;
int: IMAGES;

set of int: rows = 1..ROWS;
set of int: columns = 1..COLUMNS;
set of int: images = 1..IMAGES;

int: RxC = ROWS * COLUMNS ;
set of int: rxc = 1..RxC ;

array[images, rows, columns] of bool: bol_images = array3d(images, rows, columns, [ int_images[i,r,c] > 0 | i in images, r in rows, c in columns ] );

array[images, rxc] of bool: flattened_images = array2d(images, rxc, bol_images);

%int: HIDDEN_NEURONS

array[rxc] of var bool: weights;
var rxc: bias; 
var bool: invert;


array[images] of var bool: perceptron = [ (sum(bdot(weights, row(flattened_images,i), rxc)) >= bias) xor invert| i in images]; 


array[images] of var bool: correct = [ labels[i] = perceptron[i] | i in images  ];



solve maximize sum(correct);
/*
output ["correct = \(correct);\nperceptron = \(perceptron);\nweights = \(weights);\nbias = \(bias);\nflattened_images = \(flattened_images);\n"] ;
