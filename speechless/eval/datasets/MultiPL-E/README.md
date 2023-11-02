---
annotations_creators:
- machine-generated
language:
- en
language_creators:
- machine-generated
- expert-generated
license:
- mit
multilinguality:
- monolingual
pretty_name: MultiPLE-E
size_categories:
- 1K<n<10K
source_datasets:
- original
- extended|openai_humaneval
- extended|mbpp
tags: []
task_categories: []
task_ids: []
dataset_info:
- config_name: cpp-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 217792
    num_examples: 161
  download_size: 248493
  dataset_size: 217792
- config_name: cpp-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 239517
    num_examples: 161
  download_size: 270773
  dataset_size: 239517
- config_name: cpp-reworded
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 239767
    num_examples: 161
  download_size: 271023
  dataset_size: 239767
- config_name: cpp-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 198566
    num_examples: 158
  download_size: 227555
  dataset_size: 198566
- config_name: cs-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 259874
    num_examples: 158
  download_size: 291137
  dataset_size: 259874
- config_name: cs-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 283738
    num_examples: 158
  download_size: 315563
  dataset_size: 283738
- config_name: cs-reworded
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 283673
    num_examples: 158
  download_size: 315498
  dataset_size: 283673
- config_name: cs-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 237663
    num_examples: 155
  download_size: 267251
  dataset_size: 237663
- config_name: d-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 175592
    num_examples: 156
  download_size: 209568
  dataset_size: 175592
- config_name: d-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 181121
    num_examples: 156
  download_size: 215649
  dataset_size: 181121
- config_name: d-reworded
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 181296
    num_examples: 156
  download_size: 215824
  dataset_size: 181296
- config_name: d-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 157938
    num_examples: 153
  download_size: 190211
  dataset_size: 157938
- config_name: go-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 241130
    num_examples: 154
  download_size: 280424
  dataset_size: 241130
- config_name: go-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 247448
    num_examples: 154
  download_size: 287275
  dataset_size: 247448
- config_name: go-reworded
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 247354
    num_examples: 154
  download_size: 287181
  dataset_size: 247354
- config_name: go-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 221519
    num_examples: 151
  download_size: 258980
  dataset_size: 221519
- config_name: java-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 259836
    num_examples: 158
  download_size: 291099
  dataset_size: 259836
- config_name: java-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 286548
    num_examples: 158
  download_size: 318373
  dataset_size: 286548
- config_name: java-reworded
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 288031
    num_examples: 158
  download_size: 319856
  dataset_size: 288031
- config_name: java-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 237672
    num_examples: 155
  download_size: 267260
  dataset_size: 237672
- config_name: jl-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 163708
    num_examples: 159
  download_size: 198696
  dataset_size: 163708
- config_name: jl-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 167969
    num_examples: 159
  download_size: 203514
  dataset_size: 167969
- config_name: jl-reworded
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 168251
    num_examples: 159
  download_size: 203796
  dataset_size: 168251
- config_name: jl-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 145913
    num_examples: 156
  download_size: 179158
  dataset_size: 145913
- config_name: js-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 177635
    num_examples: 161
  download_size: 211822
  dataset_size: 177635
- config_name: js-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 181987
    num_examples: 161
  download_size: 216729
  dataset_size: 181987
- config_name: js-reworded
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 182171
    num_examples: 161
  download_size: 216913
  dataset_size: 182171
- config_name: js-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 158619
    num_examples: 158
  download_size: 191028
  dataset_size: 158619
- config_name: lua-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 180398
    num_examples: 161
  download_size: 212511
  dataset_size: 180398
- config_name: lua-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 184763
    num_examples: 161
  download_size: 216595
  dataset_size: 184763
- config_name: lua-reworded
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 184853
    num_examples: 161
  download_size: 216685
  dataset_size: 184853
- config_name: lua-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 161339
    num_examples: 158
  download_size: 191690
  dataset_size: 161339
- config_name: php-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 219526
    num_examples: 161
  download_size: 256134
  dataset_size: 219526
- config_name: php-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 225575
    num_examples: 161
  download_size: 262738
  dataset_size: 225575
- config_name: php-reworded
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 225730
    num_examples: 161
  download_size: 262893
  dataset_size: 225730
- config_name: php-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 200047
    num_examples: 158
  download_size: 234848
  dataset_size: 200047
- config_name: pl-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 239874
    num_examples: 161
  download_size: 279351
  dataset_size: 239874
- config_name: pl-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 243611
    num_examples: 161
  download_size: 283767
  dataset_size: 243611
- config_name: pl-reworded
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 243661
    num_examples: 161
  download_size: 283817
  dataset_size: 243661
- config_name: pl-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 220817
    num_examples: 158
  download_size: 258463
  dataset_size: 220817
- config_name: py-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 173537
    num_examples: 161
  download_size: 207009
  dataset_size: 173537
- config_name: py-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 177787
    num_examples: 161
  download_size: 210975
  dataset_size: 177787
- config_name: py-reworded
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 177787
    num_examples: 161
  download_size: 210975
  dataset_size: 177787
- config_name: py-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 155389
    num_examples: 158
  download_size: 187068
  dataset_size: 155389
- config_name: r-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 186803
    num_examples: 161
  download_size: 215857
  dataset_size: 186803
- config_name: r-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 191732
    num_examples: 161
  download_size: 220505
  dataset_size: 191732
- config_name: r-reworded
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 191747
    num_examples: 161
  download_size: 220520
  dataset_size: 191747
- config_name: r-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 168422
    num_examples: 158
  download_size: 195771
  dataset_size: 168422
- config_name: rb-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 181999
    num_examples: 161
  download_size: 216186
  dataset_size: 181999
- config_name: rb-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 188317
    num_examples: 161
  download_size: 223059
  dataset_size: 188317
- config_name: rb-reworded
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 188457
    num_examples: 161
  download_size: 223199
  dataset_size: 188457
- config_name: rb-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 163569
    num_examples: 158
  download_size: 195978
  dataset_size: 163569
- config_name: rkt-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 177757
    num_examples: 161
  download_size: 212266
  dataset_size: 177757
- config_name: rkt-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 182937
    num_examples: 161
  download_size: 218001
  dataset_size: 182937
- config_name: rkt-reworded
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 182754
    num_examples: 161
  download_size: 217818
  dataset_size: 182754
- config_name: rkt-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 158729
    num_examples: 158
  download_size: 191454
  dataset_size: 158729
- config_name: rs-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 177191
    num_examples: 156
  download_size: 206604
  dataset_size: 177191
- config_name: rs-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 188587
    num_examples: 156
  download_size: 218555
  dataset_size: 188587
- config_name: rs-reworded
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 188841
    num_examples: 156
  download_size: 218809
  dataset_size: 188841
- config_name: rs-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 158191
    num_examples: 153
  download_size: 185991
  dataset_size: 158191
- config_name: scala-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 222118
    num_examples: 160
  download_size: 253027
  dataset_size: 222118
- config_name: scala-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 240540
    num_examples: 160
  download_size: 272012
  dataset_size: 240540
- config_name: scala-reworded
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 240466
    num_examples: 160
  download_size: 271938
  dataset_size: 240466
- config_name: scala-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 200261
    num_examples: 157
  download_size: 229477
  dataset_size: 200261
- config_name: sh-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 158460
    num_examples: 158
  download_size: 193268
  dataset_size: 158460
- config_name: sh-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 164552
    num_examples: 158
  download_size: 201631
  dataset_size: 164552
- config_name: sh-reworded
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 164521
    num_examples: 158
  download_size: 201600
  dataset_size: 164521
- config_name: sh-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 140720
    num_examples: 155
  download_size: 173767
  dataset_size: 140720
- config_name: swift-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 201798
    num_examples: 161
  download_size: 233903
  dataset_size: 201798
- config_name: swift-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 204760
    num_examples: 158
  download_size: 236660
  dataset_size: 204760
- config_name: swift-reworded
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 204920
    num_examples: 158
  download_size: 236820
  dataset_size: 204920
- config_name: swift-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 181681
    num_examples: 158
  download_size: 212047
  dataset_size: 181681
- config_name: ts-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 181763
    num_examples: 159
  download_size: 215589
  dataset_size: 181763
- config_name: ts-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 186037
    num_examples: 159
  download_size: 220423
  dataset_size: 186037
- config_name: ts-reworded
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 186215
    num_examples: 159
  download_size: 220601
  dataset_size: 186215
- config_name: ts-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 162881
    num_examples: 156
  download_size: 194985
  dataset_size: 162881
- config_name: cpp
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 239767
    num_examples: 161
  download_size: 271023
  dataset_size: 239767
- config_name: cs
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 283673
    num_examples: 158
  download_size: 315498
  dataset_size: 283673
- config_name: d
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 181296
    num_examples: 156
  download_size: 215824
  dataset_size: 181296
- config_name: go
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 247354
    num_examples: 154
  download_size: 287181
  dataset_size: 247354
- config_name: java
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 288031
    num_examples: 158
  download_size: 319856
  dataset_size: 288031
- config_name: jl
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 168251
    num_examples: 159
  download_size: 203796
  dataset_size: 168251
- config_name: js
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 182171
    num_examples: 161
  download_size: 216913
  dataset_size: 182171
- config_name: lua
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 184853
    num_examples: 161
  download_size: 216685
  dataset_size: 184853
- config_name: php
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 225730
    num_examples: 161
  download_size: 262893
  dataset_size: 225730
- config_name: pl
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 243661
    num_examples: 161
  download_size: 283817
  dataset_size: 243661
- config_name: py
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 177787
    num_examples: 161
  download_size: 210975
  dataset_size: 177787
- config_name: r
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 191747
    num_examples: 161
  download_size: 220520
  dataset_size: 191747
- config_name: rb
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 188457
    num_examples: 161
  download_size: 223199
  dataset_size: 188457
- config_name: rkt
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 182754
    num_examples: 161
  download_size: 217818
  dataset_size: 182754
- config_name: rs
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 188841
    num_examples: 156
  download_size: 218809
  dataset_size: 188841
- config_name: scala
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 240466
    num_examples: 160
  download_size: 271938
  dataset_size: 240466
- config_name: sh
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 164521
    num_examples: 158
  download_size: 201600
  dataset_size: 164521
- config_name: swift
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 204920
    num_examples: 158
  download_size: 236820
  dataset_size: 204920
- config_name: ts
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 186215
    num_examples: 159
  download_size: 220601
  dataset_size: 186215
- config_name: humaneval-cpp-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 218990
    num_examples: 161
  download_size: 249691
  dataset_size: 218990
- config_name: humaneval-cpp-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 240786
    num_examples: 161
  download_size: 272042
  dataset_size: 240786
- config_name: humaneval-cpp
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 241036
    num_examples: 161
  download_size: 272292
  dataset_size: 241036
- config_name: humaneval-cpp-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 199746
    num_examples: 158
  download_size: 228735
  dataset_size: 199746
- config_name: humaneval-cs-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 260822
    num_examples: 158
  download_size: 292085
  dataset_size: 260822
- config_name: humaneval-cs-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 284686
    num_examples: 158
  download_size: 316511
  dataset_size: 284686
- config_name: humaneval-cs
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 284621
    num_examples: 158
  download_size: 316446
  dataset_size: 284621
- config_name: humaneval-cs-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 238593
    num_examples: 155
  download_size: 268181
  dataset_size: 238593
- config_name: humaneval-d-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 176864
    num_examples: 156
  download_size: 210856
  dataset_size: 176864
- config_name: humaneval-d-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 182057
    num_examples: 156
  download_size: 216585
  dataset_size: 182057
- config_name: humaneval-d
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 182232
    num_examples: 156
  download_size: 216760
  dataset_size: 182232
- config_name: humaneval-d-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 158856
    num_examples: 153
  download_size: 191129
  dataset_size: 158856
- config_name: humaneval-go-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 242054
    num_examples: 154
  download_size: 281348
  dataset_size: 242054
- config_name: humaneval-go-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 248372
    num_examples: 154
  download_size: 288199
  dataset_size: 248372
- config_name: humaneval-go
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 248278
    num_examples: 154
  download_size: 288105
  dataset_size: 248278
- config_name: humaneval-go-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 222425
    num_examples: 151
  download_size: 259886
  dataset_size: 222425
- config_name: humaneval-java-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 261057
    num_examples: 158
  download_size: 292320
  dataset_size: 261057
- config_name: humaneval-java-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 287860
    num_examples: 158
  download_size: 319685
  dataset_size: 287860
- config_name: humaneval-java
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 289343
    num_examples: 158
  download_size: 321168
  dataset_size: 289343
- config_name: humaneval-java-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 238875
    num_examples: 155
  download_size: 268463
  dataset_size: 238875
- config_name: humaneval-jl-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 164664
    num_examples: 159
  download_size: 199654
  dataset_size: 164664
- config_name: humaneval-jl-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 168925
    num_examples: 159
  download_size: 204472
  dataset_size: 168925
- config_name: humaneval-jl
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 169207
    num_examples: 159
  download_size: 204754
  dataset_size: 169207
- config_name: humaneval-jl-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 146851
    num_examples: 156
  download_size: 180098
  dataset_size: 146851
- config_name: humaneval-js-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 178601
    num_examples: 161
  download_size: 212788
  dataset_size: 178601
- config_name: humaneval-js-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 182953
    num_examples: 161
  download_size: 217695
  dataset_size: 182953
- config_name: humaneval-js
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 183137
    num_examples: 161
  download_size: 217879
  dataset_size: 183137
- config_name: humaneval-js-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 159567
    num_examples: 158
  download_size: 191976
  dataset_size: 159567
- config_name: humaneval-lua-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 181364
    num_examples: 161
  download_size: 213477
  dataset_size: 181364
- config_name: humaneval-lua-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 185729
    num_examples: 161
  download_size: 217561
  dataset_size: 185729
- config_name: humaneval-lua
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 185819
    num_examples: 161
  download_size: 217651
  dataset_size: 185819
- config_name: humaneval-lua-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 162287
    num_examples: 158
  download_size: 192638
  dataset_size: 162287
- config_name: humaneval-php-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 220492
    num_examples: 161
  download_size: 257100
  dataset_size: 220492
- config_name: humaneval-php-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 226541
    num_examples: 161
  download_size: 263704
  dataset_size: 226541
- config_name: humaneval-php
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 226696
    num_examples: 161
  download_size: 263859
  dataset_size: 226696
- config_name: humaneval-php-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 200995
    num_examples: 158
  download_size: 235796
  dataset_size: 200995
- config_name: humaneval-pl-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 240840
    num_examples: 161
  download_size: 280317
  dataset_size: 240840
- config_name: humaneval-pl-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 244577
    num_examples: 161
  download_size: 284733
  dataset_size: 244577
- config_name: humaneval-pl
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 244627
    num_examples: 161
  download_size: 284783
  dataset_size: 244627
- config_name: humaneval-pl-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 221765
    num_examples: 158
  download_size: 259411
  dataset_size: 221765
- config_name: humaneval-py-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 174503
    num_examples: 161
  download_size: 207975
  dataset_size: 174503
- config_name: humaneval-py-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 178753
    num_examples: 161
  download_size: 211941
  dataset_size: 178753
- config_name: humaneval-py
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 178753
    num_examples: 161
  download_size: 211941
  dataset_size: 178753
- config_name: humaneval-py-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 156337
    num_examples: 158
  download_size: 188016
  dataset_size: 156337
- config_name: humaneval-r-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 186140
    num_examples: 161
  download_size: 215194
  dataset_size: 186140
- config_name: humaneval-r-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 190637
    num_examples: 161
  download_size: 219410
  dataset_size: 190637
- config_name: humaneval-r
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 190652
    num_examples: 161
  download_size: 219425
  dataset_size: 190652
- config_name: humaneval-r-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 167742
    num_examples: 158
  download_size: 195091
  dataset_size: 167742
- config_name: humaneval-rb-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 182965
    num_examples: 161
  download_size: 217152
  dataset_size: 182965
- config_name: humaneval-rb-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 189283
    num_examples: 161
  download_size: 224025
  dataset_size: 189283
- config_name: humaneval-rb
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 189423
    num_examples: 161
  download_size: 224165
  dataset_size: 189423
- config_name: humaneval-rb-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 164517
    num_examples: 158
  download_size: 196926
  dataset_size: 164517
- config_name: humaneval-rkt-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 185503
    num_examples: 161
  download_size: 220012
  dataset_size: 185503
- config_name: humaneval-rkt-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 190683
    num_examples: 161
  download_size: 225747
  dataset_size: 190683
- config_name: humaneval-rkt
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 190500
    num_examples: 161
  download_size: 225564
  dataset_size: 190500
- config_name: humaneval-rkt-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 166379
    num_examples: 158
  download_size: 199104
  dataset_size: 166379
- config_name: humaneval-rs-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 178127
    num_examples: 156
  download_size: 207540
  dataset_size: 178127
- config_name: humaneval-rs-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 189523
    num_examples: 156
  download_size: 219491
  dataset_size: 189523
- config_name: humaneval-rs
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 189777
    num_examples: 156
  download_size: 219745
  dataset_size: 189777
- config_name: humaneval-rs-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 159109
    num_examples: 153
  download_size: 186909
  dataset_size: 159109
- config_name: humaneval-scala-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 223078
    num_examples: 160
  download_size: 253987
  dataset_size: 223078
- config_name: humaneval-scala-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 241500
    num_examples: 160
  download_size: 272972
  dataset_size: 241500
- config_name: humaneval-scala
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 241426
    num_examples: 160
  download_size: 272898
  dataset_size: 241426
- config_name: humaneval-scala-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 201203
    num_examples: 157
  download_size: 230419
  dataset_size: 201203
- config_name: humaneval-sh-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 159408
    num_examples: 158
  download_size: 194216
  dataset_size: 159408
- config_name: humaneval-sh-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 165500
    num_examples: 158
  download_size: 202579
  dataset_size: 165500
- config_name: humaneval-sh
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 165469
    num_examples: 158
  download_size: 202548
  dataset_size: 165469
- config_name: humaneval-sh-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 141650
    num_examples: 155
  download_size: 174697
  dataset_size: 141650
- config_name: humaneval-swift-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 202764
    num_examples: 161
  download_size: 234869
  dataset_size: 202764
- config_name: humaneval-swift-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 205708
    num_examples: 158
  download_size: 237608
  dataset_size: 205708
- config_name: humaneval-swift
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 205868
    num_examples: 158
  download_size: 237768
  dataset_size: 205868
- config_name: humaneval-swift-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 182629
    num_examples: 158
  download_size: 212995
  dataset_size: 182629
- config_name: humaneval-ts-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 182717
    num_examples: 159
  download_size: 216543
  dataset_size: 182717
- config_name: humaneval-ts-transform
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 186991
    num_examples: 159
  download_size: 221377
  dataset_size: 186991
- config_name: humaneval-ts
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 187169
    num_examples: 159
  download_size: 221555
  dataset_size: 187169
- config_name: humaneval-ts-remove
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 163817
    num_examples: 156
  download_size: 195921
  dataset_size: 163817
- config_name: mbpp-cpp-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 360057
    num_examples: 397
  download_size: 428174
  dataset_size: 360057
- config_name: mbpp-cpp
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 362541
    num_examples: 397
  download_size: 430658
  dataset_size: 362541
- config_name: mbpp-cs-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 416276
    num_examples: 386
  download_size: 484875
  dataset_size: 416276
- config_name: mbpp-cs
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 418156
    num_examples: 386
  download_size: 486755
  dataset_size: 418156
- config_name: mbpp-d-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 232820
    num_examples: 358
  download_size: 303807
  dataset_size: 232820
- config_name: mbpp-d
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 234776
    num_examples: 358
  download_size: 305763
  dataset_size: 234776
- config_name: mbpp-go-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 399157
    num_examples: 374
  download_size: 486803
  dataset_size: 399157
- config_name: mbpp-go
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 400841
    num_examples: 374
  download_size: 488487
  dataset_size: 400841
- config_name: mbpp-java-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 419406
    num_examples: 386
  download_size: 488005
  dataset_size: 419406
- config_name: mbpp-java
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 423652
    num_examples: 386
  download_size: 492251
  dataset_size: 423652
- config_name: mbpp-jl-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 228259
    num_examples: 390
  download_size: 305322
  dataset_size: 228259
- config_name: mbpp-jl
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 230672
    num_examples: 390
  download_size: 307735
  dataset_size: 230672
- config_name: mbpp-js-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 256499
    num_examples: 397
  download_size: 333225
  dataset_size: 256499
- config_name: mbpp-js
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 258734
    num_examples: 397
  download_size: 335460
  dataset_size: 258734
- config_name: mbpp-lua-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 262378
    num_examples: 397
  download_size: 335520
  dataset_size: 262378
- config_name: mbpp-lua
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 264635
    num_examples: 397
  download_size: 337777
  dataset_size: 264635
- config_name: mbpp-php-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 308918
    num_examples: 397
  download_size: 388541
  dataset_size: 308918
- config_name: mbpp-php
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 311263
    num_examples: 397
  download_size: 390886
  dataset_size: 311263
- config_name: mbpp-pl-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 321045
    num_examples: 396
  download_size: 402353
  dataset_size: 321045
- config_name: mbpp-pl
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 323224
    num_examples: 396
  download_size: 404532
  dataset_size: 323224
- config_name: mbpp-py-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 253037
    num_examples: 397
  download_size: 330230
  dataset_size: 253037
- config_name: mbpp-py
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 255022
    num_examples: 397
  download_size: 332215
  dataset_size: 255022
- config_name: mbpp-r-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 257698
    num_examples: 397
  download_size: 323297
  dataset_size: 257698
- config_name: mbpp-r
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 259514
    num_examples: 397
  download_size: 325113
  dataset_size: 259514
- config_name: mbpp-rb-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 266702
    num_examples: 397
  download_size: 343428
  dataset_size: 266702
- config_name: mbpp-rb
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 268881
    num_examples: 397
  download_size: 345607
  dataset_size: 268881
- config_name: mbpp-rkt-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 269019
    num_examples: 397
  download_size: 346539
  dataset_size: 269019
- config_name: mbpp-rkt
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 270933
    num_examples: 397
  download_size: 348453
  dataset_size: 270933
- config_name: mbpp-rs-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 218020
    num_examples: 354
  download_size: 277268
  dataset_size: 218020
- config_name: mbpp-rs
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 220113
    num_examples: 354
  download_size: 279361
  dataset_size: 220113
- config_name: mbpp-scala-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 330435
    num_examples: 396
  download_size: 399451
  dataset_size: 330435
- config_name: mbpp-scala
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 332677
    num_examples: 396
  download_size: 401693
  dataset_size: 332677
- config_name: mbpp-sh-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 217246
    num_examples: 382
  download_size: 289241
  dataset_size: 217246
- config_name: mbpp-sh
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 219035
    num_examples: 382
  download_size: 291030
  dataset_size: 219035
- config_name: mbpp-swift-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 317271
    num_examples: 396
  download_size: 388726
  dataset_size: 317271
- config_name: mbpp-swift
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 319946
    num_examples: 396
  download_size: 391401
  dataset_size: 319946
- config_name: mbpp-ts-keep
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 265973
    num_examples: 390
  download_size: 341007
  dataset_size: 265973
- config_name: mbpp-ts
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 268179
    num_examples: 390
  download_size: 343213
  dataset_size: 268179
---

# Dataset Card for MultiPL-E

## Dataset Description

- **Homepage:**  https://nuprl.github.io/MultiPL-E/
- **Repository:**  https://github.com/nuprl/MultiPL-E
- **Paper:** https://ieeexplore.ieee.org/abstract/document/10103177
- **Point of Contact:** carolyn.anderson@wellesley.edu, mfeldman@oberlin.edu, a.guha@northeastern.edu

## Dataset Summary

MultiPL-E is a dataset for evaluating large language models for code
generation that supports 18 programming languages. It takes the OpenAI 
"HumanEval" and the MBPP Python benchmarks and uses little compilers to
translate them  to other languages. It is easy to add support for new languages 
and benchmarks.

## Subsets

For most purposes, you should use the variations called *SRCDATA-LANG*, where
*SRCDATA* is either "humaneval" or "mbpp" and *LANG* is one of the supported
languages. We use the canonical file extension for each language to identify
the language, e.g., "py" for Python, "cpp" for C++, "lua" for Lua, and so on.

We also provide a few other variations:

- *SRCDATA-LANG-keep* is the same as *SRCDATA-LANG*, but the text of the prompt
  is totally unchanged. If the original prompt had Python doctests, they remain
  as Python instead of being translated to *LANG*. If the original prompt had 
  Python-specific terminology, e.g., "list", it remains "list", instead of 
  being translated, e.g., to "vector" for C++.

- *SRCDATA-LANG-transform* transforms the doctests to *LANG* but leaves
  the natural language text of the prompt unchanged.

- *SRCDATA-LANG-removed* removes the doctests from the prompt.

Note that MBPP does not have any doctests, so the "removed" and "transform"
variations are not available for MBPP.

## Example

The following script uses the Salesforce/codegen model to generate Lua
and MultiPL-E to produce a script with unit tests for luaunit.

```python
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM

LANG = "lua"
MODEL_NAME = "Salesforce/codegen-350M-multi"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).half().cuda()
problems = datasets.load_dataset("nuprl/MultiPL-E", f"humaneval-{LANG}")

def stop_at_stop_token(decoded_string, problem):
    """
    Truncates the output at stop tokens, taking care to skip the prompt
    which may have stop tokens.
    """
    min_stop_index = len(decoded_string)
    for stop_token in problem["stop_tokens"]:
        stop_index = decoded_string.find(stop_token)
        if stop_index != -1 and stop_index > len(problem["prompt"]) and stop_index < min_stop_index:
            min_stop_index = stop_index
    return decoded_string[:min_stop_index]

for problem in problems["test"]:
    input_ids = tokenizer(
        problem["prompt"],
        return_tensors="pt",
    ).input_ids.cuda()
    generated_ids = model.generate(
        input_ids, max_length=512, pad_token_id=tokenizer.eos_token_id + 2
    )
    truncated_string = stop_at_stop_token(tokenizer.decode(generated_ids[0]), problem)
    filename = problem["name"] + "." + LANG
    with open(filename, "w") as f:
        print(f"Created {filename}")
        f.write(truncated_string)
        f.write("\n")
        f.write(problem["tests"])
```