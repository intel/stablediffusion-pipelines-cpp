// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#pragma once

#include <regex>
#include <unordered_map>
#include <map>
#include <string>
#include <list>
#include <vector>
#include <stdio.h>
#include <optional>
#include <stdexcept>
#include <set>
#include <limits>

namespace cpp_stable_diffusion_ov
{
    //todo: throw this into some kind of utility header
    static inline std::vector<int64_t> vslice(std::vector<int64_t>& in_v,
        std::optional<int64_t> start,
        std::optional<int64_t> end)
    {
        std::vector<int64_t>::iterator s;
        if (!start)
        {
            s = in_v.begin();
        }
        else if (*start >= 0)
        {
            if (*start > (int64_t)in_v.size())
                return std::vector<int64_t>();
            else
                s = in_v.begin() + *start;
        }
        else
        {
            if (-(*start) > (int64_t)in_v.size())
                s = in_v.begin();
            else
                s = in_v.begin() + (in_v.size() + *start);
        }

        std::vector<int64_t>::iterator e;
        if (!end)
        {
            e = in_v.end();
        }
        else if (*end >= 0)
        {
            if (*end > (int64_t)in_v.size())
                e = in_v.end();
            else
                e = in_v.begin() + *end;
        }
        else
        {
            if (-(*end) > (int64_t)in_v.size())
                e = s;
            else
                e = in_v.begin() + (in_v.size() + *end);
        }

        if (s > e)
            return std::vector<int64_t>();

        return std::vector<int64_t>(s, e);
    }

    enum class PaddingStrategy { LONGEST, MAX_LENGTH, DO_NOT_PAD };

    // Wrapper class to try to replicate the python tokenizer base class support 
    // of padding parameter as either a bool, a string, or PaddingStrategy
    class Padding
    {
    public:

        Padding(bool vbal)
        {
            _bval = vbal;
        }

        Padding(const char* pad_string0)
        {
            std::string pad_string = pad_string0;
            if (pad_string == "longest")
            {
                _strategy = PaddingStrategy::LONGEST;
            }
            else if (pad_string == "max_length")
            {
                _strategy = PaddingStrategy::MAX_LENGTH;
            }
            else if (pad_string == "do_not_pad")
            {
                _strategy = PaddingStrategy::DO_NOT_PAD;
            }
            else
            {
                throw std::invalid_argument("Padding string must be 'longest', 'max_length', or 'do_not_pad'");
            }
        }

        Padding(PaddingStrategy strategy)
        {
            _strategy = strategy;
        }

        //if this padding *is* a bool, and it's true
        bool is_true()
        {
            if (_bval && (*_bval))
                return true;

            return false;
        }

        //if this padding *is* a bool, and it's false
        bool is_false()
        {
            if (_bval && !(*_bval))
                return true;

            return false;
        }

        bool operator==(const PaddingStrategy& strategy)
        {
            if (_strategy && *_strategy == strategy)
                return true;

            return false;
        }

        std::optional< bool  > as_bool() { return _bval; };
        std::optional< PaddingStrategy  > as_strategy() { return _strategy; };

    private:
        std::optional< bool  > _bval = {};
        std::optional< PaddingStrategy > _strategy = {};
    };


    enum class TruncationStrategy { ONLY_FIRST, ONLY_SECOND, LONGEST_FIRST, DO_NOT_TRUNCATE };

    // Wrapper class to try to replicate the python tokenizer base class support 
    // of truncation parameter as either a bool, a string, or TruncationStrategy
    class Truncation
    {
    public:

        Truncation(bool vbal)
        {
            _bval = vbal;

            if (!_bval)
                _strategy = TruncationStrategy::DO_NOT_TRUNCATE;
        }

        Truncation(const char* trunc_string0)
        {
            std::string trunc_string = trunc_string0;
            if (trunc_string == "only_first")
            {
                _strategy = TruncationStrategy::ONLY_FIRST;
            }
            else if (trunc_string == "only_second")
            {
                _strategy = TruncationStrategy::ONLY_SECOND;
            }
            else if (trunc_string == "longest_first")
            {
                _strategy = TruncationStrategy::LONGEST_FIRST;
            }
            else if (trunc_string == "do_not_truncate")
            {
                _strategy = TruncationStrategy::DO_NOT_TRUNCATE;
            }
            else
            {
                throw std::invalid_argument("Padding string must be 'longest', 'max_length', or 'do_not_pad'");
            }
        }

        Truncation(TruncationStrategy strategy)
        {
            _strategy = strategy;
        }

        //if this truncation *is* a bool, and it's true
        bool is_true()
        {
            if (_bval && (*_bval))
                return true;

            return false;
        }

        //if this truncation *is* a bool, and it's false
        bool is_false()
        {
            if (_bval && !(*_bval))
                return true;

            return false;
        }

        bool operator==(const TruncationStrategy& strategy)
        {
            if (_strategy && *_strategy == strategy)
                return true;

            return false;
        }

        std::optional< bool  > as_bool() { return _bval; };
        std::optional< TruncationStrategy  > as_strategy() { return _strategy; };

    private:
        std::optional< bool  > _bval = {};
        std::optional< TruncationStrategy > _strategy = {};
    };

    //TODO -- make value an OpenVINO tensor?
    typedef std::unordered_map<std::string, std::vector<std::vector<int64_t>>> BatchEncoding;
    typedef std::unordered_map<std::string, std::vector<int64_t>> Batch1Encoding;

#if 0
    class BatchEncoding
    {
    public:
        BatchEncoding();

        std::vector<int64_t>& operator[](std::string si)
        {
            return _encodings[si];
        }

    private:

        std::unordered_map<std::string, std::vector<int64_t>> _encodings;
    };
#endif


    class Trie
    {

    public:

        struct TrieNode
        {

        public:

            std::map<char, TrieNode*> _children;
            bool _terminated = false;
        };

        ~Trie()
        {
            for (auto* child : _allocated)
            {
                delete child;
            }
        }

        //Passes over every char (utf-8 char) on word and recursively adds it to the internal `data` trie representation.
        //The special key `""` is used to represent termination.

        //This function is idempotent, adding twice the same word will leave the trie unchanged
        void add(const std::string& word)
        {
            if (word.empty())
            {
                //Prevent empty string
                return;
            }

            TrieNode* ref = &_root;

            for (char c : word)
            {
                std::map<char, TrieNode*>::iterator it = ref->_children.find(c);
                if (it == ref->_children.end())
                {
                    TrieNode* child = new TrieNode();
                    ref->_children[c] = child;
                    _allocated.push_back(child);
                    ref = child;
                }
                else
                {
                    ref = it->second;
                }
            }

            ref->_terminated = true;
        }

        //Will look for the words added to the trie within `text`. Output is the original string splitted along the
        //boundaries of the words found.

        //This trie will match the longest possible word first !
        std::list< std::string> split(const std::string text)
        {

            // States are going to capture every possible start(indexes as above)
            // as keys, and have as values, a pointer to the position in the trie
            // where we're at. This is a partial match for now.
            // This enables to keep track of multiple matches while we're iterating
            // the string
            // If the trie contains, "blowing", and "lower" and we encounter the
            // string "blower", we need to split into["b", "lower"].
            // This is where we need to keep track of multiple possible starts.
            std::map< int64_t, TrieNode*> states;

            // This will contain every indices where we need
            // to cut.
            // We force to cut at offset 0 and len(text) (added later)
            std::list<int64_t> offsets = { 0 };

            // This is used by the lookahead which needs to skip over
            // some text where the full match exceeded the place in the initial
            // for loop
            int64_t skip = 0;

            // Main loop, Giving this algorithm O(n) complexity
            int64_t current = 0;

            for (int64_t current = 0; current < text.length(); current++)
            {
                char current_char = text[current];
                if (skip && current < skip)
                {
                    //Prevents the lookahead for matching twice
                    // like extra_id_100 and id_100
                    continue;
                }

                // This will track every state 
                // that stop matching, we need to stop tracking them.
                // If we look at "lowball", we're going to match "l" (add it to states), "o", "w", then
                // fail on "b", we need to remove 0 from the valid states.
                std::set<int64_t> to_remove;

                // Whenever we found a match, we need to drop everything
                // this is a greedy algorithm, it will match on the first found token
                bool reset = false;

                // In this case, we already have partial matches (But unfinished)
                auto states_right_now0 = states; // <- to match python behavior, create deep copy and iterate over that,
                                                //    since states is modified within the loop.
                for (auto& items : states_right_now0)
                {
                    int64_t start = items.first;
                    TrieNode* trie_pointer = items.second;
                    if (trie_pointer->_terminated)
                    {
                        int64_t end = 0;
                        // This is a final match, we need to resetand
                        // store the results in `offsets`.

                        // Lookahead to match longest first
                        // Important in case of extra_id_1 vs extra_id_100
                        // Here we are also actively looking for other earlier partial
                        // matches
                        // "[CLS]", "L", we need to match CLS even if L is special
                        for (auto& items1 : states)
                        {
                            int64_t lookstart = items1.first;
                            TrieNode* looktrie_pointer = items1.second;

                            int64_t lookahead_index = 0;
                            if (lookstart > start)
                            {
                                //This partial match is later, we can stop looking
                                break;
                            }
                            else if (lookstart < start)
                            {
                                // This partial match is earlier, the trie pointer
                                // was already updated, so index is + 1
                                lookahead_index = current + 1;
                                end = current + 1;
                            }
                            else
                            {
                                // Here lookstart == start and
                                //      looktrie_pointer == trie_pointer
                                // It wasn't updated yet so indices are current ones
                                lookahead_index = current;
                                end = current;
                            }

                            //next_char = text[lookahead_index] if lookahead_index < len(text) else None
                            std::optional<char> next_char = {};
                            if (lookahead_index < text.length())
                            {
                                next_char = text[lookahead_index];
                            }

                            if (looktrie_pointer->_terminated)
                            {
                                start = lookstart;
                                end = lookahead_index;
                                skip = lookahead_index;
                            }

                            if (next_char)
                            {
                                std::map<char, TrieNode*>::iterator it = looktrie_pointer->_children.find(*next_char);
                                while (it != looktrie_pointer->_children.end())
                                {
                                    looktrie_pointer = it->second;
                                    lookahead_index += 1;
                                    if (looktrie_pointer->_terminated)
                                    {
                                        start = lookstart;
                                        end = lookahead_index;
                                        skip = lookahead_index;
                                    }

                                    if (lookahead_index == text.length())
                                    {
                                        // End of string
                                        break;
                                    }

                                    next_char = text[lookahead_index];
                                    it = looktrie_pointer->_children.find(*next_char);
                                } //End lookahead
                            }
                        }

                        // Storing and resetting
                        offsets.push_back(start);
                        offsets.push_back(end);
                        reset = true;
                        break;
                    }
                    else if (trie_pointer->_children.count(current_char))
                    {
                        // The current character being looked at has a match within the trie
                        // update the pointer (it will be stored back into states later).
                        trie_pointer = trie_pointer->_children[current_char];

                        // Storing back the new pointer into the states.
                        // Partial matches got longer by one.
                        //hmm, we added a entry to state, which we're currently looping over. Check this.
                        states[start] = trie_pointer;
                    }
                    else
                    {
                        // The new character has not match in the trie, we need
                        // to stop keeping track of this partial match.
                        // We can't do it directly within the loop because of how
                        // python iteration works
                        to_remove.insert(start);
                    }
                }

                // Either clearing the full start (we found a real match)
                // Or clearing only the partial matches that didn't work.
                if (reset)
                {
                    states = {};
                }
                else
                {
                    for (auto start : to_remove)
                    {
                        states.erase(start);
                    }
                }

                // If this character is a starting character within the trie
                // start keeping track of this partial match.
                if (current >= skip && _root._children.count(current_char))
                {
                    states[current] = _root._children[current_char];
                }
            }

            // We have a cut at the end with states.
            for (auto& items : states)
            {
                int64_t start = items.first;
                TrieNode* trie_pointer = items.second;

                if (trie_pointer->_terminated)
                {
                    // This is a final match, we need to reset and
                    // store the results in offsets
                    int64_t end = text.length();
                    offsets.push_back(start);
                    offsets.push_back(end);
                    // Longest cut is always the one with lower start so the first
                    // item so we need to break.
                    break;
                }
            }

            return cut_text(text, offsets);
        }

        std::list<std::string> cut_text(const std::string& text, std::list<int64_t>& offsets)
        {
            // We have all the offsets now, we just need to do the actual splitting.
            // We need to eventually add the first part of the stringand the eventual
            // last part.
            offsets.push_back(text.length());
            std::list < std::string > tokens;
            int64_t start = 0;

            for (auto end : offsets)
            {
                if (start > end)
                {
                    fprintf(stderr, "Error! There was a bug in Trie algorithm in tokenization. Attempting to recover. Please report it anyway.");
                    continue;
                }
                else if (start == end)
                {
                    // This might happen if there's a match at index 0
                    // we're also preventing zero-width cuts in case of two
                    // consecutive matches
                    continue;
                }

                tokens.push_back(text.substr(start, end - start));
                start = end;
            }

            return tokens;
        }

        //todo:
        /*
        def _is_whitespace(char):
            """Checks whether `char` is a whitespace character."""
            # \t, \n, and \r are technically control characters but we treat them
            # as whitespace since they are generally considered as such.
            if char == " " or char == "\t" or char == "\n" or char == "\r":
                return True
            cat = unicodedata.category(char)
            if cat == "Zs":
                return True
            return False


        def _is_control(char):
            """Checks whether `char` is a control character."""
            # These are technically control characters but we count them as whitespace
            # characters.
            if char == "\t" or char == "\n" or char == "\r":
                return False
            cat = unicodedata.category(char)
            if cat.startswith("C"):
                return True
            return False


        def _is_punctuation(char):
            """Checks whether `char` is a punctuation character."""
            cp = ord(char)
            # We treat all non-letter/number ASCII as punctuation.
            # Characters such as "^", "$", and "`" are not in the Unicode
            # Punctuation class but we treat them as punctuation anyways, for
            # consistency.
            if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
                return True
            cat = unicodedata.category(char)
            if cat.startswith("P"):
                return True
            return False


        def _is_end_of_word(text):
            """Checks whether the last character in text is one of a punctuation, control or whitespace character."""
            last_char = text[-1]
            return bool(_is_control(last_char) | _is_punctuation(last_char) | _is_whitespace(last_char))


        def _is_start_of_word(text):
            """Checks whether the first character in text is one of a punctuation, control or whitespace character."""
            first_char = text[0]
            return bool(_is_control(first_char) | _is_punctuation(first_char) | _is_whitespace(first_char))


        def _insert_one_token_to_ordered_list(token_list: List[str], new_token: str):
            """
            Inserts one token to an ordered list if it does not already exist. Note: token_list must be sorted.
            """
            insertion_idx = bisect.bisect_left(token_list, new_token)
            # Checks if new_token is already in the ordered token_list
            if insertion_idx < len(token_list) and token_list[insertion_idx] == new_token:
                # new_token is in token_list, don't add
                return
            else:
                token_list.insert(insertion_idx, new_token)
        */



    private:

        TrieNode _root;
        std::list< TrieNode*> _allocated;
    };


    //std::lower_bound(a, x) is equivelant to  bisect.bisect_left(a, x)

    static void _insert_one_token_to_ordered_list(std::list<std::string>& token_list, const std::string& new_token)
    {
        auto insertion_idx = std::lower_bound(token_list.begin(), token_list.end(), new_token);
        if ((insertion_idx != token_list.end()) && (*insertion_idx == new_token))
        {
            //new_token is in token_list, don't add
            return;
        }
        else
        {
            token_list.insert(insertion_idx, new_token);
        }
    }

    static inline void to_lower(std::string& str)
    {
        std::transform(str.begin(), str.end(), str.begin(),
            [](unsigned char c) { return std::tolower(c); });
    }


    // trim from left
    inline std::string& ltrim(std::string& s, const char* t = " \t\n\r\f\v")
    {
        s.erase(0, s.find_first_not_of(t));
        return s;
    }

    // trim from right
    inline std::string& rtrim(std::string& s, const char* t = " \t\n\r\f\v")
    {
        s.erase(s.find_last_not_of(t) + 1);
        return s;
    }
}