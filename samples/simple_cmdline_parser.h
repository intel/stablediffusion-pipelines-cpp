// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#pragma once
#include <string>
#include <set>
#include <optional>
#include <map>
#include <stdexcept>
#include <iostream>

class SimpleCmdLineParser
{
public:

    SimpleCmdLineParser(int argc, char* argv[])
    {
        //for this simple parser, all arguments should start with "--". So, basically we only
        // accept the following two forms
        // --some_key=some_val
        // or
        // --some_key 
        std::string prefix = "--";

        for (int i = 1; i < argc; i++)
        {
            std::string argv_str = argv[i];

            if (argv_str.compare(0, prefix.size(), prefix) != 0)
            {
                throw std::invalid_argument("All arguments must be prefixed with '--'");
            }

            size_t ci = 0;
            bool eq_found = false;
            for (char c : argv_str)
            {
                if (c == '=')
                {
                    eq_found = true;
                    break;
                }

                 ci++;
            }

            std::string::iterator end_it = argv_str.end();
            if (eq_found)
                end_it = argv_str.begin() + ci;

            std::string key = std::string(argv_str.begin() + 2, end_it);


            if (_keys.count(key) != 0)
            {
                throw std::invalid_argument("Duplicate argument given (--" + key + ")");
            }

            _keys.insert(key);

            //std::cout << "found key = " << key << std::endl;

            if (end_it != argv_str.end())
            {
                //std::cout << "ci = " << ci << std::endl;
                std::string value = std::string(argv_str.begin() + ci + 1, argv_str.end());
                //std::cout << "value for key " << key <<  " = " << value << std::endl;
                _key_to_val[key] = value;
            }
        }   
    }

    bool is_help_needed()
    {
        return (zero_args_given() || was_key_given("h") || was_key_given("help"));
    }

    bool zero_args_given() { return _keys.empty(); };

    bool was_key_given(const std::string& key)
    {
        return (_keys.count(key) > 0);
    }

    std::optional<std::string> get_value_for_key(const std::string& key)
    {
        std::optional<std::string> ret = {};

        if (_key_to_val.count(key) > 0)
            ret = _key_to_val[key];

        return ret;
    }


private:

    std::set< std::string > _keys;
    std::map< std::string, std::string > _key_to_val;

};

