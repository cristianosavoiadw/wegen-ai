#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace engine {

// Tokenizer simplificado para Fase 3
// Versão completa: integrar SentencePiece ou Tiktoken
class SimpleTokenizer {
public:
    SimpleTokenizer() = default;

    // Carrega vocabulário do modelo GGUF
    bool load_from_gguf(const std::string& model_path);

    // Tokeniza texto → IDs
    std::vector<int32_t> encode(const std::string& text) const;

    // Detokeniza IDs → texto
    std::string decode(const std::vector<int32_t>& tokens) const;

    // Tokens especiais
    int32_t bos_token() const { return bos_token_; }
    int32_t eos_token() const { return eos_token_; }
    int32_t pad_token() const { return pad_token_; }

    // Tamanho do vocabulário
    size_t vocab_size() const { return vocab_.size(); }

private:
    std::vector<std::string> vocab_;
    int32_t bos_token_ = 1;  // <s>
    int32_t eos_token_ = 2;  // </s>
    int32_t pad_token_ = 0;  // <pad>

    // Fallback: tokenização por espaço (TEMPORÁRIO)
    std::vector<int32_t> encode_whitespace(const std::string& text) const;
};

} // namespace engine