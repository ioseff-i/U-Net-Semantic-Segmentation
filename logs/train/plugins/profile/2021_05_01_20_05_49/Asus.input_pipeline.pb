	h?>c?@h?>c?@!h?>c?@	?"?dj???"?dj??!?"?dj??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$h?>c?@g?v???AU?g$?`?@Y??j?#???*	g??|??S@2U
Iterator::Model::ParallelMapV2w?$???!?	?ǡ4@)w?$???1?	?ǡ4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???V???!??jEz8@)??N?`???1.?i??3@:Preprocessing2F
Iterator::Modelp?4(???!vA???C@)?:?? ???1)?kV?53@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??i??!??@???7@)e????`??1?????1@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??7L4??!???TN@)zpw?n???1R?v#'?$@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?VBwI?u?!?.?R$?@)?VBwI?u?1?.?R$?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor? ?X4?m?!,k?u?f@)? ?X4?m?1,k?u?f@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap ??X??!I?V??H9@)n??T?1g?_a-???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?"?dj??I?٬??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	g?v???g?v???!g?v???      ??!       "      ??!       *      ??!       2	U?g$?`?@U?g$?`?@!U?g$?`?@:      ??!       B      ??!       J	??j?#?????j?#???!??j?#???R      ??!       Z	??j?#?????j?#???!??j?#???b      ??!       JCPU_ONLYY?"?dj??b q?٬??X@