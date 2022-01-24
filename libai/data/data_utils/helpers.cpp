/*
 coding=utf-8
 Copyright 2021 The OneFlow Authors. All rights reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

/* Helper methods for fast index mapping builds */

#include <algorithm>
#include <iostream>
#include <limits>
#include <math.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <random>
#include <stdexcept>

namespace py = pybind11;
using namespace std;

const int32_t LONG_SENTENCE_LEN = 512;


py::array build_sample_idx(const py::array_t<int64_t> &doc_idx_,
                           const py::array_t<int64_t> &sizes_,
                           const int32_t seq_length, 
                           const int64_t num_tokens) {
  /* Sample index (sample_idx) is used for gpt2 like dataset for which
     the documents are flattened and the samples are built based on this
     1-D flatten array. It is a 2D array with sizes [number-of-samples + 1, 2]
     where [..., 0] contains the index into `doc_idx` and [..., 1] is the
     starting offset in that document.*/
  // 对于所有的文档，根据最大句子长度计算每条样本的初始文档序号，初始相对偏移量，最终文档序号，最终相对偏移量。
  // 由于可能出现一篇文档结束，但样本长度不足最大句子长度的情形，此时对应初始文档与最终文档序号不同，然后遍历
  // 中间的所有文档，然后把它们拼接成一个样本。
  // Consistency checks.
  assert(seq_length > 1);
  assert(num_tokens > 1);

  // Remove bound checks.
  auto sizes = sizes_.unchecked<1>();
  auto doc_idx = doc_idx_.unchecked<1>();

  // Mapping and it's length (1D).
  int64_t num_samples = (num_tokens - 1) / seq_length;
  int64_t *sample_idx = new int64_t[2 * (num_samples + 1)];

  cout << "    using:" << endl << std::flush;
  cout << "     number of documents:       " << doc_idx_.shape(0) - 1
       << endl
       << std::flush;
  cout << "     sequence length:           " << seq_length << endl
       << std::flush;
  cout << "     total number of samples:   " << num_samples << endl
       << std::flush;

  // Index into sample_idx.
  int64_t sample_index = 0;
  // Index into doc_idx.
  int64_t doc_idx_index = 0;
  // Begining offset for each document.
  int64_t doc_offset = 0;
  // Start with first document and no offset.
  sample_idx[2 * sample_index] = doc_idx_index;
  sample_idx[2 * sample_index + 1] = doc_offset;
  ++sample_index;

  while (sample_index <= num_samples) {
    // Start with a fresh sequence.
    int64_t remaining_seq_length = seq_length + 1;
    while (remaining_seq_length != 0) {
      // Get the document length.
      auto doc_id = doc_idx[doc_idx_index];
      auto doc_length = sizes[doc_id] - doc_offset;
      // And add it to the current sequence.
      remaining_seq_length -= doc_length;
      // If we have more than a full sequence, adjust offset and set
      // remaining length to zero so we return from the while loop.
      // Note that -1 here is for the same reason we have -1 in
      // `_num_epochs` calculations.
      if (remaining_seq_length <= 0) {
        doc_offset += (remaining_seq_length + doc_length - 1);
        remaining_seq_length = 0;
      } else {
        // Otherwise, start from the begining of the next document.
        ++doc_idx_index;
        doc_offset = 0;
      }
    }
    // Record the sequence.
    sample_idx[2 * sample_index] = doc_idx_index;
    sample_idx[2 * sample_index + 1] = doc_offset;
    ++sample_index;
  }

  // Method to deallocate memory.
  py::capsule free_when_done(sample_idx, [](void *mem_) {
    int64_t *mem = reinterpret_cast<int64_t *>(mem_);
    delete[] mem;
  });

  // Return the numpy array.
  const auto byte_size = sizeof(int64_t);
  return py::array(std::vector<int64_t>{num_samples + 1, 2}, // shape
                   {2 * byte_size, byte_size}, // C-style contiguous strides
                   sample_idx,                 // the data pointer
                   free_when_done);            // numpy array references
}

inline int32_t get_target_sample_len(
    const int32_t short_seq_ratio, 
    const int32_t max_seq_length, 
    std::mt19937& rand32_gen) {
  if (short_seq_ratio == 0) {
    return max_seq_length;
  }
  const auto random_number = rand32_gen();
  if ((random_number % short_seq_ratio) == 0) {
    return 2 + random_number % (max_seq_length - 1);
  }
  return max_seq_length;
}

template <typename DocIdx>
py::array build_mapping_impl(
    const py::array_t<int64_t> &docs_,
    const py::array_t<int64_t> &sizes_, 
    const int32_t max_seq_length,
    const double short_seq_prob,
    const bool verbose, 
    const int32_t min_num_sent) {
  /* Build a mapping of (start-index, end-index, sequence-length) where
     start and end index are the indices of the sentences in the sample
     and sequence-length is the target sequence length.
  */

  // Consistency checks.
  assert(max_seq_length > 1);
  assert(short_seq_prob >= 0.0);
  assert(short_seq_prob <= 1.0);

  // Remove bound checks.
  auto docs = docs_.unchecked<1>();
  auto sizes = sizes_.unchecked<1>();

  // For efficiency, convert probability to ratio. Note: rand() generates int.
  int32_t short_seq_ratio = 0;
  if (short_seq_prob > 0) {
    short_seq_ratio = static_cast<int32_t>(round(1.0 / short_seq_prob));
  }

  if (verbose) {
    const auto sent_start_index = docs[0];
    const auto sent_end_index = docs[docs_.shape(0) - 1];
    const auto num_sentences = sent_end_index - sent_start_index;
    cout << "    using:" << endl << std::flush;
    cout << "     number of documents:            " << docs_.shape(0) - 1
         << endl
         << std::flush;
    cout << "     sentences range:                [" << sent_start_index << ", "
         << sent_end_index << ")" << endl
         << std::flush;
    cout << "     total number of sentences:      " << num_sentences << endl
         << std::flush;
    cout << "     maximum sequence length:        " << max_seq_length << endl
         << std::flush;
    cout << "     short sequence probability:     " << short_seq_prob << endl
         << std::flush;
    cout << "     short sequence ratio (1/prob):  " << short_seq_ratio << endl
         << std::flush;
  }

  // Mapping and it's length (1D).
  int64_t num_samples = -1;
  DocIdx *maps = NULL;

  // Perform two iterations, in the first iteration get the size
  // and allocate memory and in the second iteration populate the map.
  bool second = false;
  for (int32_t iteration = 0; iteration < 2; ++iteration) {

    // todo(dangkai): we set seed as a constant value.
    std::mt19937 rand32_gen(42);

    // Set the flag on second iteration.
    second = (iteration == 1);

    // Counters:
    uint64_t empty_docs = 0;
    uint64_t one_sent_docs = 0;
    uint64_t long_sent_docs = 0;

    // Current map index.
    uint64_t map_index = 0;

    // For each document:
    for (int32_t doc = 0; doc < (docs.shape(0) - 1); ++doc) {

      // Document sentences are in [sent_index_first, sent_index_last)
      const auto sent_index_first = docs[doc];
      const auto sent_index_last = docs[doc + 1];

      // At the begining of the document previous index is the
      // start index.
      auto prev_start_index = sent_index_first;

      // Remaining documents.
      auto num_remain_sent = sent_index_last - sent_index_first;

      // Some bookkeeping
      if (!second) {
        if (num_remain_sent == 0) {
          ++empty_docs;
        }
        if (num_remain_sent == 1) {
          ++one_sent_docs;
        }
      }

      // Detect documents with long sentences.
      bool contains_long_sentence = false;
      if (num_remain_sent > 1) {
        for (auto sent_index = sent_index_first; sent_index < sent_index_last;
              ++sent_index) {
          if (sizes[sent_index] > LONG_SENTENCE_LEN) {
            if (!second) {
              ++long_sent_docs;
            }
            contains_long_sentence = true;
            break;
          }
        }
      }

      // If we have more than two sentences.
      if ((num_remain_sent >= min_num_sent) && (!contains_long_sentence)) {

        // Set values.
        auto seq_len = int64_t{0};
        auto num_sent = int64_t{0};
        auto target_seq_len = get_target_sample_len(short_seq_ratio, max_seq_length, rand32_gen);

        // Loop through sentences.
        for (auto sent_index = sent_index_first; sent_index < sent_index_last;
              ++sent_index) {

          // Add the size and number of sentences.
          seq_len += sizes[sent_index];
          ++num_sent;
          --num_remain_sent;

          // If we have reached the target length.
          // and if not only one sentence is left in the document.
          // and if we have at least two sentneces.
          // and if we have reached end of the document.
          if (((seq_len >= target_seq_len) && (num_remain_sent > 1) &&
                (num_sent >= min_num_sent)) ||
              (num_remain_sent == 0)) {

            // Check for overflow.
            if ((3 * map_index + 2) > std::numeric_limits<int64_t>::max()) {
              cout << "number of samples exceeded maximum "
                    << "allowed by type int64: "
                    << std::numeric_limits<int64_t>::max() << endl;
              throw std::overflow_error("Number of samples");
            }

            // Populate the map.
            if (second) {
              const auto map_index_0 = 3 * map_index;
              maps[map_index_0] = static_cast<DocIdx>(prev_start_index);
              maps[map_index_0 + 1] = static_cast<DocIdx>(sent_index + 1);
              maps[map_index_0 + 2] = static_cast<DocIdx>(target_seq_len);
            }

            // Update indices / counters.
            ++map_index;
            prev_start_index = sent_index + 1;
            target_seq_len = get_target_sample_len(short_seq_ratio, max_seq_length, rand32_gen);
            seq_len = 0;
            num_sent = 0;
          }
        } // for (auto sent_index=sent_index_first; ...
      }   // if (num_remain_sent > 1) {
    }     // for (int doc=0; doc < num_docs; ++doc) {

    if (!second) {
      if (verbose) {
        cout << "   number of empty documents: " << empty_docs << endl
             << std::flush;
        cout << "   number of documents with one sentence: " << one_sent_docs
             << endl
             << std::flush;
        cout << "   number of documents with long sentences: " << long_sent_docs
             << endl
             << std::flush;
        cout << "   will create mapping for " << map_index << " samples" << endl
             << std::flush;
      }
      assert(maps == NULL);
      assert(num_samples < 0);
      maps = new DocIdx[3 * map_index];
      num_samples = static_cast<int64_t>(map_index);
    }

  } // for (int iteration=0; iteration < 2; ++iteration) {

  // Method to deallocate memory.
  py::capsule free_when_done(maps, [](void *mem_) {
    DocIdx *mem = reinterpret_cast<DocIdx *>(mem_);
    delete[] mem;
  });

  // Return the numpy array.
  const auto byte_size = sizeof(DocIdx);
  return py::array(std::vector<int64_t>{num_samples, 3}, // shape
                   {3 * byte_size, byte_size}, // C-style contiguous strides
                   maps,                       // the data pointer
                   free_when_done);            // numpy array references
}

py::array build_mapping(const py::array_t<int64_t> &docs_,
                        const py::array_t<int64_t> &sizes_,
                        const int max_seq_length,
                        const double short_seq_prob,
                        const bool verbose,
                        const int32_t min_num_sent) {

  if (sizes_.size() > std::numeric_limits<uint32_t>::max()) {
    if (verbose) {
      cout << "    using uint64 for data mapping..." << endl << std::flush;
    }
    return build_mapping_impl<uint64_t>(docs_, sizes_, max_seq_length, short_seq_prob, verbose, min_num_sent);
  } else {
    if (verbose) {
      cout << "    using uint32 for data mapping..." << endl << std::flush;
    }
    return build_mapping_impl<uint32_t>(docs_, sizes_, max_seq_length, short_seq_prob, verbose, min_num_sent);
  }
}


PYBIND11_MODULE(helpers, m) {
  m.def("build_mapping", &build_mapping);
  m.def("build_sample_idx", &build_sample_idx);
}