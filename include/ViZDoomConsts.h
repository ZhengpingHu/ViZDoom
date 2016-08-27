/*
 Copyright (C) 2016 by Wojciech Jaśkowski, Michał Kempka, Grzegorz Runc, Jakub Toczek, Marek Wydmuch

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
*/

#ifndef __VIZDOOM_DEFINES_H__
#define __VIZDOOM_DEFINES_H__

namespace vizdoom{

    #define VIZDOOM_LIB_VERSION 109
    #define VIZDOOM_LIB_VERSION_STR "1.1.0pre"

    const unsigned int SLOT_COUNT = 10;
    const unsigned int MAX_PLAYERS = 8;
    const unsigned int MAX_PLAYER_NAME_LENGTH = 16;
    const unsigned int USER_VARIABLE_COUNT = 30;
    const unsigned int DEFAULT_TICRATE = 35;

    const unsigned int BINARY_BUTTON_COUNT = 38;
    const unsigned int DELTA_BUTTON_COUNT = 5;
    const unsigned int BUTTON_COUNT = 43;

}
#endif
