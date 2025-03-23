#pragma once

#include "Storage.hpp"
#include "Typing.hpp"
#include <cassert>
#include <cmath>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct MMapFile {
  DT *m_data;
  u32 m_byte_size;

  MMapFile(MMapFile const &) = delete;
  MMapFile(MMapFile &&) = delete;

  explicit MMapFile(std::string const &name) {
    struct File {
      u32 m_fd;
      explicit File(const char *name) : m_fd(open(name, O_RDONLY)) {
        if (m_fd == -1) {
          throw std::runtime_error("Failed to open file: " + std::string(name));
        }
      }
      ~File() {
        if (m_fd != -1)
          close(m_fd);
      }
    };
    File file(name.c_str());

    struct stat sb;
    if (fstat(file.m_fd, &sb) == -1) {
      throw std::runtime_error("Failed to get file size");
    }
    m_byte_size = sb.st_size;
    std::cout << "load " + name << ": " << m_byte_size << " bytes\n";
    m_data = (DT *)mmap(nullptr, m_byte_size, PROT_READ, MAP_PRIVATE, file.m_fd, 0);
    if (m_data == MAP_FAILED) {
      throw std::runtime_error("Memory mapping failed");
    }
  }
  ~MMapFile() {
    if (m_data != nullptr)
      munmap(m_data, m_byte_size);
  }

  using value_type = DT;
  value_type const &operator[](u32 index) const { return m_data[index]; }
  value_type &operator[](u32 index) { return m_data[index]; }

  static std::shared_ptr<MMapFile> load(std::string const &path) { return std::make_shared<MMapFile>(path); }
};

static_assert(IMatrixStorage<MMapFile>, "should fulfill");
