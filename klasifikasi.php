<?php

/**
 * harus di require setiap kali menggunakan library
 * dari composer
 */
require_once('vendor/autoload.php');

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;

// Load dataset dari csv
$dataset = Labeled::fromIterator(new CSV('car.csv',true));

//dataset sebelum diubah ke numeric
// print_r($dataset);

//ubah dataset string ke numeric
$transformer = new NumericStringConverter();
$dataset->apply($transformer);

//dataset setelah diubah ke numeric
// print_r($dataset);

[$training, $testing] = $dataset->stratifiedSplit(0.8);

$estimator = new KNearestNeighbors(3);

$estimator->train($training);

// cek apakah proses training berhasil
// var_dump($estimator->trained());

// testing menggunakan 1 data saja
// $single_test = [
//     [2, 6, 2005]
// ];

// $contoh = new Unlabeled($single_test);

// testing menggunakan satu data
// $predictions = $estimator->predict($contoh);

//hasil testing 1 data
// print_r($predictions);

// testing menggunakan data testing dari dataset
$predictions = $estimator->predict($testing);


//hasil testing menggunakan data testing dari dataset
// print_r($predictions);


// validasi, confusion matrix, akurasi, dll
$report = new MulticlassBreakdown();

$results = $report->generate($predictions, $testing->labels());

echo $results;