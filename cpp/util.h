// Copyright 2016 Raymond Yun Fei, Christopher Batty, Robert Bridson
//
// Licensed under the Apache License,
// Version 2.0(the "License");
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

#ifndef APIC2D_UTIL_H_
#define APIC2D_UTIL_H_

#include <algorithm>
#include <iostream>
#include <vector>
using scalar = float;

#ifndef M_PI
const double M_PI = 3.1415926535897932384626433832795;
#endif

#ifdef WIN32
#undef min
#undef max
#endif

using std::max;
using std::min;
using std::swap;

template <class T>
inline T sqr(const T& x) {
  return x * x;
}

template <class T>
inline T cube(const T& x) {
  return x * x * x;
}

template <class T>
inline T min(T a1, T a2, T a3) {
  return min(a1, min(a2, a3));
}

template <class T>
inline T min(T a1, T a2, T a3, T a4) {
  return min(min(a1, a2), min(a3, a4));
}

template <class T>
inline T min(T a1, T a2, T a3, T a4, T a5) {
  return min(min(a1, a2), min(a3, a4), a5);
}

template <class T>
inline T min(T a1, T a2, T a3, T a4, T a5, T a6) {
  return min(min(a1, a2), min(a3, a4), min(a5, a6));
}

template <class T>
inline T max(T a1, T a2, T a3) {
  return max(a1, max(a2, a3));
}

template <class T>
inline T max(T a1, T a2, T a3, T a4) {
  return max(max(a1, a2), max(a3, a4));
}

template <class T>
inline T max(T a1, T a2, T a3, T a4, T a5) {
  return max(max(a1, a2), max(a3, a4), a5);
}

template <class T>
inline T max(T a1, T a2, T a3, T a4, T a5, T a6) {
  return max(max(a1, a2), max(a3, a4), max(a5, a6));
}

template <class T>
inline void minmax(T a1, T a2, T& amin, T& amax) {
  if (a1 < a2) {
    amin = a1;
    amax = a2;
  } else {
    amin = a2;
    amax = a1;
  }
}

template <class T>
inline void minmax(T a1, T a2, T a3, T& amin, T& amax) {
  if (a1 < a2) {
    if (a1 < a3) {
      amin = a1;
      if (a2 < a3)
        amax = a3;
      else
        amax = a2;
    } else {
      amin = a3;
      if (a1 < a2)
        amax = a2;
      else
        amax = a1;
    }
  } else {
    if (a2 < a3) {
      amin = a2;
      if (a1 < a3)
        amax = a3;
      else
        amax = a1;
    } else {
      amin = a3;
      amax = a1;
    }
  }
}

template <class T>
inline void minmax(T a1, T a2, T a3, T a4, T& amin, T& amax) {
  if (a1 < a2) {
    if (a3 < a4) {
      amin = min(a1, a3);
      amax = max(a2, a4);
    } else {
      amin = min(a1, a4);
      amax = max(a2, a3);
    }
  } else {
    if (a3 < a4) {
      amin = min(a2, a3);
      amax = max(a1, a4);
    } else {
      amin = min(a2, a4);
      amax = max(a1, a3);
    }
  }
}

template <class T>
inline void minmax(T a1, T a2, T a3, T a4, T a5, T& amin, T& amax) {
  //@@@ the logic could be shortcircuited a lot!
  amin = min(a1, a2, a3, a4, a5);
  amax = max(a1, a2, a3, a4, a5);
}

template <class T>
inline void minmax(T a1, T a2, T a3, T a4, T a5, T a6, T& amin, T& amax) {
  //@@@ the logic could be shortcircuited a lot!
  amin = min(a1, a2, a3, a4, a5, a6);
  amax = max(a1, a2, a3, a4, a5, a6);
}

template <class T>
inline void update_minmax(T a1, T& amin, T& amax) {
  if (a1 < amin)
    amin = a1;
  else if (a1 > amax)
    amax = a1;
}

template <class T>
inline void sort(T& a, T& b, T& c) {
  T temp;
  if (a < b) {
    if (a < c) {
      if (c < b) {  // a<c<b
        temp = c;
        c = b;
        b = temp;
      }       // else: a<b<c
    } else {  // c<a<b
      temp = c;
      c = b;
      b = a;
      a = temp;
    }
  } else {
    if (b < c) {
      if (a < c) {  // b<a<c
        temp = b;
        b = a;
        a = temp;
      } else {  // b<c<a
        temp = b;
        b = c;
        c = a;
        a = temp;
      }
    } else {  // c<b<a
      temp = c;
      c = a;
      a = temp;
    }
  }
}

template <class T>
inline T clamp(T a, T lower, T upper) {
  if (a < lower)
    return lower;
  else if (a > upper)
    return upper;
  else
    return a;
}

// only makes sense with T=float or double
template <class T>
inline T smooth_step(T r) {
  if (r < 0)
    return 0;
  else if (r > 1)
    return 1;
  return r * r * r * (10 + r * (-15 + r * 6));
}

// only makes sense with T=float or double
template <class T>
inline T smooth_step(T r, T r_lower, T r_upper, T value_lower, T value_upper) {
  return value_lower + smooth_step((r - r_lower) / (r_upper - r_lower)) * (value_upper - value_lower);
}

// only makes sense with T=float or double
template <class T>
inline T ramp(T r) {
  return smooth_step((r + 1) / 2) * 2 - 1;
}

inline unsigned int round_up_to_power_of_two(unsigned int n) {
  int exponent = 0;
  --n;
  while (n) {
    ++exponent;
    n >>= 1;
  }
  return 1 << exponent;
}

inline unsigned int round_down_to_power_of_two(unsigned int n) {
  int exponent = 0;
  while (n > 1) {
    ++exponent;
    n >>= 1;
  }
  return 1 << exponent;
}

// Transforms even the sequence 0,1,2,3,... into reasonably good random numbers
// Challenge: improve on this in speed and "randomness"!
// This seems to pass several statistical tests, and is a bijective map (of
// 32-bit unsigned ints)
inline unsigned int randhash(unsigned int seed) {
  unsigned int i = (seed ^ 0xA3C59AC3u) * 2654435769u;
  i ^= (i >> 16);
  i *= 2654435769u;
  i ^= (i >> 16);
  i *= 2654435769u;
  return i;
}

// the inverse of randhash
inline unsigned int unhash(unsigned int h) {
  h *= 340573321u;
  h ^= (h >> 16);
  h *= 340573321u;
  h ^= (h >> 16);
  h *= 340573321u;
  h ^= 0xA3C59AC3u;
  return h;
}

template <class T>
void zero(std::vector<T>& v) {
  for (int i = (int)v.size() - 1; i >= 0; --i) v[i] = 0;
}

template <class T>
T abs_max(const std::vector<T>& v) {
  T m = 0;
  for (int i = (int)v.size() - 1; i >= 0; --i) {
    if (std::fabs(v[i]) > m) m = std::fabs(v[i]);
  }
  return m;
}

template <class T>
bool contains(const std::vector<T>& a, T e) {
  for (unsigned int i = 0; i < a.size(); ++i)
    if (a[i] == e) return true;
  return false;
}

template <class T>
void add_unique(std::vector<T>& a, T e) {
  for (unsigned int i = 0; i < a.size(); ++i)
    if (a[i] == e) return;
  a.push_back(e);
}

template <class T>
void insert(std::vector<T>& a, unsigned int index, T e) {
  a.push_back(a.back());
  for (unsigned int i = (unsigned int)a.size() - 1; i > index; --i) a[i] = a[i - 1];
  a[index] = e;
}

template <class T>
void erase(std::vector<T>& a, unsigned int index) {
  for (unsigned int i = index; i < a.size() - 1; ++i) a[i] = a[i + 1];
  a.pop_back();
}

template <class T>
void erase_swap(std::vector<T>& a, unsigned int index) {
  for (unsigned int i = index; i < a.size() - 1; ++i) swap(a[i], a[i + 1]);
  a.pop_back();
}

template <class T>
void erase_unordered(std::vector<T>& a, unsigned int index) {
  a[index] = a.back();
  a.pop_back();
}

template <class T>
void erase_unordered_swap(std::vector<T>& a, unsigned int index) {
  swap(a[index], a.back());
  a.pop_back();
}

template <class T>
void find_and_erase_unordered(std::vector<T>& a, const T& doomed_element) {
  for (unsigned int i = 0; i < a.size(); ++i)
    if (a[i] == doomed_element) {
      erase_unordered(a, i);
      return;
    }
}

template <class T>
void replace_once(std::vector<T>& a, const T& old_element, const T& new_element) {
  for (unsigned int i = 0; i < a.size(); ++i)
    if (a[i] == old_element) {
      a[i] = new_element;
      return;
    }
}

template <class T>
void write_matlab(std::ostream& output, const std::vector<T>& a, const char* variable_name, bool column_vector = true, int significant_digits = 18) {
  output << variable_name << "=[";
  std::streamsize old_precision = output.precision();
  output.precision(significant_digits);
  for (unsigned int i = 0; i < a.size(); ++i) {
    output << a[i] << " ";
  }
  output << "]";
  if (column_vector) output << "'";
  output << ";" << std::endl;
  output.precision(old_precision);
}

#endif  // APIC2D_UTIL_H_
