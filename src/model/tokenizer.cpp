#include "model/tokenizer.h"
#include "model/gguf_loader.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_map>
#include <iostream>
#include <cstring>

#ifdef __linux__
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace engine {

// ============================================================================
// BPE (Byte Pair Encoding) Helper
// ============================================================================

struct BPEToken {
    std::string text;
    int32_t id;
    float score;
};

class BPETokenizer {
public:
    void add_token(int32_t id, const std::string& text, float score) {
        id_to_token_[id] = {text, id, score};
        token_to_id_[text] = id;
    }

    bool has_token(const std::string& text) const {
        return token_to_id_.find(text) != token_to_id_.end();
    }

    int32_t get_id(const std::string& text) const {
        auto it = token_to_id_.find(text);
        return (it != token_to_id_.end()) ? it->second : -1;
    }

    const std::string& get_text(int32_t id) const {
        static const std::string empty;
        auto it = id_to_token_.find(id);
        return (it != id_to_token_.end()) ? it->second.text : empty;
    }

    size_t vocab_size() const {
        return id_to_token_.size();
    }

    // Encode usando BPE
    std::vector<int32_t> encode_bpe(const std::string& text) const {
        std::vector<int32_t> result;

        // Fallback: tokenização character-level se não tiver merge rules
        std::string current;
        for (char c : text) {
            current += c;
            int32_t id = get_id(current);

            if (id >= 0) {
                continue; // Tenta estender o match
            } else {
                // Não encontrou match maior, volta um char
                if (current.size() > 1) {
                    current.pop_back();
                    id = get_id(current);
                    if (id >= 0) {
                        result.push_back(id);
                    }
                    current = std::string(1, c);
                } else {
                    // Single char - usa fallback
                    result.push_back(static_cast<uint8_t>(c) + 3);
                    current.clear();
                }
            }
        }

        // Flush remaining
        if (!current.empty()) {
            int32_t id = get_id(current);
            if (id >= 0) {
                result.push_back(id);
            } else {
                // Character fallback
                for (char c : current) {
                    result.push_back(static_cast<uint8_t>(c) + 3);
                }
            }
        }

        return result;
    }

private:
    std::unordered_map<int32_t, BPEToken> id_to_token_;
    std::unordered_map<std::string, int32_t> token_to_id_;
};

// ============================================================================
// SimpleTokenizer Implementation
// ============================================================================

struct TokenizerImpl {
    BPETokenizer bpe;
    int32_t bos = 1;
    int32_t eos = 2;
    int32_t pad = 0;
    int32_t unk = 3;
    bool loaded = false;
};

SimpleTokenizer::SimpleTokenizer()
    : impl_(std::make_unique<TokenizerImpl>()) {
}

SimpleTokenizer::~SimpleTokenizer() = default;

bool SimpleTokenizer::load_from_gguf(const std::string& model_path) {
    std::cout << "[tokenizer] loading vocabulary from GGUF...\n";

#ifndef __linux__
    std::cerr << "[tokenizer] mmap only on Linux\n";
    return load_fallback();
#else
    int fd = ::open(model_path.c_str(), O_RDONLY);
    if (fd < 0) {
        std::cerr << "[tokenizer] cannot open file\n";
        return load_fallback();
    }

    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        return load_fallback();
    }

    void* base = ::mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);

    if (base == MAP_FAILED) {
        return load_fallback();
    }

    bool success = parse_gguf_vocab(base, st.st_size);
    ::munmap(base, st.st_size);

    if (!success) {
        std::cerr << "[tokenizer] failed to parse GGUF vocab\n";
        return load_fallback();
    }

    impl_->loaded = true;
    std::cout << "[tokenizer] loaded " << impl_->bpe.vocab_size() << " tokens\n";
    return true;
#endif
}

bool SimpleTokenizer::parse_gguf_vocab(void* base, size_t size) {
    const uint8_t* p = static_cast<const uint8_t*>(base);
    const uint8_t* end = p + size;

    // Verifica magic
    if (size < 8 || std::memcmp(p, "GGUF", 4) != 0) {
        return false;
    }
    p += 4;

    // Version
    uint32_t version;
    std::memcpy(&version, p, 4);
    p += 4;

    if (version < 2 || version > 3) {
        return false;
    }

    // Tensor count
    uint64_t n_tensors;
    std::memcpy(&n_tensors, p, 8);
    p += 8;

    // KV count
    uint64_t n_kv;
    std::memcpy(&n_kv, p, 8);
    p += 8;

    // Parse KV pairs looking for tokenizer data
    for (uint64_t i = 0; i < n_kv && p < end; ++i) {
        // Read key
        uint64_t key_len;
        std::memcpy(&key_len, p, 8);
        p += 8;

        if (p + key_len > end) break;

        std::string key(reinterpret_cast<const char*>(p), key_len);
        p += key_len;

        // Read type
        if (p + 4 > end) break;
        uint32_t vtype;
        std::memcpy(&vtype, p, 4);
        p += 4;

        // Process tokenizer.ggml.tokens (array of strings)
        if (key == "tokenizer.ggml.tokens" && vtype == 9) { // ARRAY
            if (!parse_token_array(p, end)) {
                skip_value(p, end, vtype);
            }
        }
        // Process tokenizer.ggml.scores (array of floats)
        else if (key == "tokenizer.ggml.scores" && vtype == 9) {
            // TODO: parse scores
            skip_value(p, end, vtype);
        }
        // Special tokens
        else if (key == "tokenizer.ggml.bos_token_id" && vtype == 4) {
            std::memcpy(&impl_->bos, p, 4);
            p += 4;
        }
        else if (key == "tokenizer.ggml.eos_token_id" && vtype == 4) {
            std::memcpy(&impl_->eos, p, 4);
            p += 4;
        }
        else if (key == "tokenizer.ggml.padding_token_id" && vtype == 4) {
            std::memcpy(&impl_->pad, p, 4);
            p += 4;
        }
        else if (key == "tokenizer.ggml.unknown_token_id" && vtype == 4) {
            std::memcpy(&impl_->unk, p, 4);
            p += 4;
        }
        else {
            skip_value(p, end, vtype);
        }
    }

    return impl_->bpe.vocab_size() > 0;
}

bool SimpleTokenizer::parse_token_array(const uint8_t*& p, const uint8_t* end) {
    if (p + 12 > end) return false;

    // Array element type (should be STRING = 8)
    uint32_t elem_type;
    std::memcpy(&elem_type, p, 4);
    p += 4;

    if (elem_type != 8) { // STRING
        return false;
    }

    // Array length
    uint64_t n_tokens;
    std::memcpy(&n_tokens, p, 8);
    p += 8;

    std::cout << "[tokenizer] found " << n_tokens << " tokens in GGUF\n";

    // Read each token string
    for (uint64_t i = 0; i < n_tokens && p < end; ++i) {
        // String length
        uint64_t str_len;
        if (p + 8 > end) break;
        std::memcpy(&str_len, p, 8);
        p += 8;

        if (p + str_len > end) break;

        std::string token(reinterpret_cast<const char*>(p), str_len);
        p += str_len;

        // Add to BPE with default score
        impl_->bpe.add_token(static_cast<int32_t>(i), token, 0.0f);
    }

    return true;
}

void SimpleTokenizer::skip_value(const uint8_t*& p, const uint8_t* end, uint32_t type) {
    switch (type) {
        case 0: case 1: case 7: // UINT8, INT8, BOOL
            if (p + 1 <= end) p += 1;
            break;
        case 2: case 3: // UINT16, INT16
            if (p + 2 <= end) p += 2;
            break;
        case 4: case 5: case 6: // UINT32, INT32, FLOAT32
            if (p + 4 <= end) p += 4;
            break;
        case 10: case 11: case 12: // UINT64, INT64, FLOAT64
            if (p + 8 <= end) p += 8;
            break;
        case 8: { // STRING
            if (p + 8 > end) break;
            uint64_t len;
            std::memcpy(&len, p, 8);
            p += 8;
            if (p + len <= end) p += len;
            break;
        }
        case 9: { // ARRAY
            if (p + 12 > end) break;
            uint32_t elem_type;
            std::memcpy(&elem_type, p, 4);
            p += 4;
            uint64_t n;
            std::memcpy(&n, p, 8);
            p += 8;
            for (uint64_t i = 0; i < n && p < end; ++i) {
                skip_value(p, end, elem_type);
            }
            break;
        }
    }
}

bool SimpleTokenizer::load_fallback() {
    std::cout << "[tokenizer] using fallback vocabulary\n";

    // Tokens especiais
    impl_->bpe.add_token(0, "<pad>", 0.0f);
    impl_->bpe.add_token(1, "<s>", 0.0f);
    impl_->bpe.add_token(2, "</s>", 0.0f);
    impl_->bpe.add_token(3, "<unk>", 0.0f);

    // Espaço
    impl_->bpe.add_token(4, " ", 0.0f);

    // Common words
    const char* common[] = {
        "the", "a", "an", "is", "are", "was", "were",
        "hello", "world", "AI", "model", "neural", "network",
        ".", ",", "?", "!", "\n"
    };

    int32_t id = 100;
    for (const char* word : common) {
        impl_->bpe.add_token(id++, word, 0.0f);
    }

    impl_->loaded = true;
    return true;
}

// ============================================================================
// ENCODE / DECODE
// ============================================================================

std::vector<int32_t> SimpleTokenizer::encode(const std::string& text) const {
    if (!impl_->loaded) {
        return encode_whitespace(text);
    }

    std::vector<int32_t> tokens;
    tokens.push_back(impl_->bos);

    // Usa BPE
    auto bpe_tokens = impl_->bpe.encode_bpe(text);
    tokens.insert(tokens.end(), bpe_tokens.begin(), bpe_tokens.end());

    tokens.push_back(impl_->eos);
    return tokens;
}

std::string SimpleTokenizer::decode(const std::vector<int32_t>& tokens) const {
    std::string result;

    for (int32_t token_id : tokens) {
        // Ignora tokens especiais
        if (token_id == impl_->bos ||
            token_id == impl_->eos ||
            token_id == impl_->pad) {
            std::cout << "[tokenizer] skipping special token: " << token_id << "\n";
            continue;
        }

        // Verifica bounds
        if (token_id < 0 || static_cast<size_t>(token_id) >= impl_->bpe.vocab_size()) {
            std::cout << "[tokenizer] WARNING: token_id " << token_id
                      << " out of range (vocab=" << impl_->bpe.vocab_size() << ")\n";
            result += "<unk>";
            continue;
        }

        const std::string& text = impl_->bpe.get_text(token_id);

        // DEBUG: Mostrar o que está sendo decodificado
        std::cout << "[tokenizer] token " << token_id << " -> '" << text << "'\n";

        if (!text.empty()) {
            result += text;
        } else {
            std::cout << "[tokenizer] WARNING: empty text for token " << token_id << "\n";
            result += "<unk>";
        }
    }

    return result;
}

std::vector<int32_t> SimpleTokenizer::encode_whitespace(const std::string& text) const {
    std::vector<int32_t> tokens;
    tokens.push_back(impl_->bos);

    std::istringstream iss(text);
    std::string word;

    while (iss >> word) {
        uint32_t hash = 0;
        for (char c : word) {
            hash = hash * 31 + static_cast<uint32_t>(c);
        }
        int32_t token_id = 100 + (hash % 30000);
        tokens.push_back(token_id);
    }

    tokens.push_back(impl_->eos);
    return tokens;
}

int32_t SimpleTokenizer::bos_token() const { return impl_->bos; }
int32_t SimpleTokenizer::eos_token() const { return impl_->eos; }
int32_t SimpleTokenizer::pad_token() const { return impl_->pad; }
size_t SimpleTokenizer::vocab_size() const { return impl_->bpe.vocab_size(); }

} // namespace engine