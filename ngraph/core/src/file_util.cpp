//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#include <ftw.h>
#include <sys/file.h>
#include <sys/time.h>
#include <unistd.h>
#endif
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#include "ngraph/env_util.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/log.hpp"

#ifdef _WIN32
#define RMDIR(a) RemoveDirectoryA(a)
#define RMFILE(a) DeleteFileA(a)
#else
#define RMDIR(a) rmdir(a)
#define RMFILE(a) remove(a)
#endif

using namespace std;
using namespace ngraph;

string file_util::get_file_name(const string& s)
{
    string rc = s;
    auto pos = s.find_last_of('/');
    if (pos != string::npos)
    {
        rc = s.substr(pos + 1);
    }
    return rc;
}

string file_util::get_file_ext(const string& s)
{
    string rc = get_file_name(s);
    auto pos = rc.find_last_of('.');
    if (pos != string::npos)
    {
        rc = rc.substr(pos);
    }
    else
    {
        rc = "";
    }
    return rc;
}

string file_util::get_directory(const string& s)
{
    string rc = s;
    auto pos = s.find_last_of('/');
    if (pos != string::npos)
    {
        rc = s.substr(0, pos);
    }
    return rc;
}

string file_util::path_join(const string& s1, const string& s2, const string& s3)
{
    return path_join(path_join(s1, s2), s3);
}

string file_util::path_join(const string& s1, const string& s2, const string& s3, const string& s4)
{
    return path_join(path_join(path_join(s1, s2), s3), s4);
}

string file_util::path_join(const string& s1, const string& s2)
{
    string rc;
    if (s2.size() > 0)
    {
        if (s2[0] == '/')
        {
            rc = s2;
        }
        else if (s1.size() > 0)
        {
            rc = s1;
            if (rc[rc.size() - 1] != '/')
            {
                rc += "/";
            }
            rc += s2;
        }
        else
        {
            rc = s2;
        }
    }
    else
    {
        rc = s1;
    }
    return rc;
}

#ifndef _WIN32
static void iterate_files_worker(const string& path,
                                 function<void(const string& file, bool is_dir)> func,
                                 bool recurse,
                                 bool include_links)
{
    DIR* dir;
    struct dirent* ent;
    if ((dir = opendir(path.c_str())) != nullptr)
    {
        try
        {
            while ((ent = readdir(dir)) != nullptr)
            {
                string name = ent->d_name;
                string path_name = file_util::path_join(path, name);
                switch (ent->d_type)
                {
                case DT_DIR:
                    if (name != "." && name != "..")
                    {
                        if (recurse)
                        {
                            file_util::iterate_files(path_name, func, recurse);
                        }
                        func(path_name, true);
                    }
                    break;
                case DT_LNK:
                    if (include_links)
                    {
                        func(path_name, false);
                    }
                    break;
                case DT_REG: func(path_name, false); break;
                default: break;
                }
            }
        }
        catch (...)
        {
            exception_ptr p = current_exception();
            closedir(dir);
            rethrow_exception(p);
        }
        closedir(dir);
    }
    else
    {
        throw runtime_error("error enumerating file " + path);
    }
}
#endif

void file_util::iterate_files(const string& path,
                              function<void(const string& file, bool is_dir)> func,
                              bool recurse,
                              bool include_links)
{
    vector<string> files;
    vector<string> dirs;
#ifdef _WIN32
    std::string file_match = path_join(path, "*");
    WIN32_FIND_DATAA data;
    HANDLE hFind = FindFirstFileA(file_match.c_str(), &data);
    if (hFind != INVALID_HANDLE_VALUE)
    {
        do
        {
            bool is_dir = data.dwFileAttributes == FILE_ATTRIBUTE_DIRECTORY;
            if (is_dir)
            {
                if (string(data.cFileName) != "." && string(data.cFileName) != "..")
                {
                    string dir_path = path_join(path, data.cFileName);
                    if (recurse)
                    {
                        iterate_files(dir_path, func, recurse);
                    }
                    func(dir_path, true);
                }
            }
            else
            {
                string file_name = path_join(path, data.cFileName);
                func(file_name, false);
            }
        } while (FindNextFileA(hFind, &data));
        FindClose(hFind);
    }
#else
    iterate_files_worker(path,
                         [&files, &dirs](const string& file, bool is_dir) {
                             if (is_dir)
                             {
                                 dirs.push_back(file);
                             }
                             else
                             {
                                 files.push_back(file);
                             }
                         },
                         recurse,
                         include_links);
#endif

    for (auto f : files)
    {
        func(f, false);
    }
    for (auto f : dirs)
    {
        func(f, true);
    }
}
