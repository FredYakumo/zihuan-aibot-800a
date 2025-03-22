#ifndef I18N_H
#define I18N_H

/**
 * Checks if a byte is the leading byte of a UTF-8 character.
 * 
 * @param c The byte to check.
 * @return True if the byte is a UTF-8 leading byte, false otherwise.
 */
#include <cstddef>
inline bool is_utf8_leader_byte(unsigned char c) {
    return (c & 0xC0) != 0x80; // Leading bytes do not start with 0b10xxxxxx.
}

/**
 * Calculates the length of a UTF-8 character based on its leading byte.
 * 
 * @param c The leading byte of the UTF-8 character.
 * @return The length of the UTF-8 character in bytes (1-4). If the byte is invalid, returns 1.
 */
inline size_t utf8_char_length(unsigned char c) {
    if ((c & 0x80) == 0x00) {
        return 1; // 0xxxxxxx: 1-byte character (ASCII).
    } else if ((c & 0xE0) == 0xC0) {
        return 2; // 110xxxxx: 2-byte character.
    } else if ((c & 0xF0) == 0xE0) {
        return 3; // 1110xxxx: 3-byte character.
    } else if ((c & 0xF8) == 0xF0) {
        return 4; // 11110xxx: 4-byte character.
    }
    return 1; // Invalid UTF-8 byte, treat as a single byte.
}

#endif