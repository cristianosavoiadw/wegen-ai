#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <memory>

namespace engine {

// Forward declaration
struct TokenizerImpl;

class SimpleTokenizer {
public:
    SimpleTokenizer();
    ~SimpleTokenizer();

    // Não copiável
    SimpleTokenizer(const SimpleTokenizer&) = delete;
    SimpleTokenizer& operator=(const SimpleTokenizer&) = delete;

    // Carrega vocabulário do modelo GGUF
    bool load_from_gguf(const std::string& model_path);

    // Tokeniza texto → IDs
    std::vector<int32_t> encode(const std::string& text) const;

    // Detokeniza IDs → texto
    std::string decode(const std::vector<int32_t>& tokens) const;

    // Tokens especiais
    int32_t bos_token() const;
    int32_t eos_token() const;
    int32_t pad_token() const;

    // Tamanho do vocabulário
    size_t vocab_size() const;

private:
    std::unique_ptr<TokenizerImpl> impl_;

    // Internal methods
    bool parse_gguf_vocab(void* base, size_t size);
    bool parse_token_array(const uint8_t*& p, const uint8_t* end);
    void skip_value(const uint8_t*& p, const uint8_t* end, uint32_t type);
    bool load_fallback();

    // Fallback: tokenização por espaço
    std::vector<int32_t> encode_whitespace(const std::string& text) const;
};

} // namespace engine