#include "am_qadf_native/processing/signal_generation.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <vector>
#include <array>
#include <string>
#include <sstream>
#include <cctype>
#include <map>
#include <functional>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace am_qadf_native {
namespace processing {

std::vector<float> SignalGeneration::generateSynthetic(
    const std::vector<std::array<float, 3>>& points,
    const std::string& signal_type,
    float amplitude,
    float frequency
) {
    if (signal_type == "gaussian") {
        // Use center of points as Gaussian center
        std::array<float, 3> center = {0.0f, 0.0f, 0.0f};
        for (const auto& p : points) {
            center[0] += p[0];
            center[1] += p[1];
            center[2] += p[2];
        }
        if (!points.empty()) {
            center[0] /= points.size();
            center[1] /= points.size();
            center[2] /= points.size();
        }
        return generateGaussian(points, center, amplitude, frequency);
    } else if (signal_type == "sine") {
        return generateSineWave(points, amplitude, frequency);
    } else if (signal_type == "random") {
        return generateRandom(points.size(), 0.0f, amplitude);
    }
    
    // Default: return zeros
    return std::vector<float>(points.size(), 0.0f);
}

std::vector<float> SignalGeneration::generateGaussian(
    const std::vector<std::array<float, 3>>& points,
    const std::array<float, 3>& center,
    float amplitude,
    float sigma
) {
    std::vector<float> values;
    values.reserve(points.size());
    
    for (const auto& p : points) {
        float dx = p[0] - center[0];
        float dy = p[1] - center[1];
        float dz = p[2] - center[2];
        float distance_sq = dx*dx + dy*dy + dz*dz;
        
        float value = amplitude * std::exp(-distance_sq / (2.0f * sigma * sigma));
        values.push_back(value);
    }
    
    return values;
}

std::vector<float> SignalGeneration::generateSineWave(
    const std::vector<std::array<float, 3>>& points,
    float amplitude,
    float frequency,
    const std::array<float, 3>& direction
) {
    std::vector<float> values;
    values.reserve(points.size());
    
    // Normalize direction
    float dir_len = std::sqrt(direction[0]*direction[0] + 
                             direction[1]*direction[1] + 
                             direction[2]*direction[2]);
    if (dir_len == 0.0f) {
        return std::vector<float>(points.size(), 0.0f);
    }
    
    float nx = direction[0] / dir_len;
    float ny = direction[1] / dir_len;
    float nz = direction[2] / dir_len;
    
    for (const auto& p : points) {
        float projection = p[0]*nx + p[1]*ny + p[2]*nz;
        float value = amplitude * std::sin(2.0f * M_PI * frequency * projection);
        values.push_back(value);
    }
    
    return values;
}

std::vector<float> SignalGeneration::generateRandom(
    size_t num_points,
    float min_value,
    float max_value,
    unsigned int seed
) {
    std::vector<float> values;
    values.reserve(num_points);
    
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(min_value, max_value);
    
    for (size_t i = 0; i < num_points; ++i) {
        values.push_back(dist(gen));
    }
    
    return values;
}

// Simple expression parser helper class
class SimpleExpressionParser {
private:
    std::string expr_;
    size_t pos_;
    
    // Skip whitespace
    void skipWhitespace() {
        while (pos_ < expr_.length() && std::isspace(expr_[pos_])) {
            pos_++;
        }
    }
    
    // Parse a number
    float parseNumber() {
        skipWhitespace();
        size_t start = pos_;
        bool has_dot = false;
        
        while (pos_ < expr_.length()) {
            if (std::isdigit(expr_[pos_])) {
                pos_++;
            } else if (expr_[pos_] == '.' && !has_dot) {
                has_dot = true;
                pos_++;
            } else {
                break;
            }
        }
        
        if (pos_ > start) {
            return std::stof(expr_.substr(start, pos_ - start));
        }
        return 0.0f;
    }
    
    // Parse a variable (x, y, z, r)
    float parseVariable(char var, float x, float y, float z) {
        skipWhitespace();
        if (pos_ < expr_.length() && expr_[pos_] == var) {
            pos_++;
            if (var == 'x') return x;
            if (var == 'y') return y;
            if (var == 'z') return z;
            if (var == 'r') return std::sqrt(x*x + y*y + z*z);
        }
        return 0.0f;
    }
    
    // Parse a function call (sin, cos, exp, log, sqrt, etc.)
    float parseFunction(float x, float y, float z) {
        skipWhitespace();
        size_t start = pos_;
        
        // Read function name
        while (pos_ < expr_.length() && std::isalpha(expr_[pos_])) {
            pos_++;
        }
        
        if (pos_ == start) {
            return 0.0f;
        }
        
        std::string func_name = expr_.substr(start, pos_ - start);
        skipWhitespace();
        
        // Expect '('
        if (pos_ >= expr_.length() || expr_[pos_] != '(') {
            return 0.0f;
        }
        pos_++;  // Skip '('
        
        // Parse argument
        float arg = parseExpression(x, y, z);
        
        skipWhitespace();
        if (pos_ < expr_.length() && expr_[pos_] == ')') {
            pos_++;  // Skip ')'
        }
        
        // Evaluate function
        if (func_name == "sin") return std::sin(arg);
        if (func_name == "cos") return std::cos(arg);
        if (func_name == "tan") return std::tan(arg);
        if (func_name == "exp") return std::exp(arg);
        if (func_name == "log") return std::log(arg);
        if (func_name == "sqrt") return std::sqrt(arg);
        if (func_name == "abs") return std::abs(arg);
        
        return 0.0f;
    }
    
    // Parse a factor (number, variable, function, or parenthesized expression)
    float parseFactor(float x, float y, float z) {
        skipWhitespace();
        if (pos_ >= expr_.length()) {
            return 0.0f;
        }
        
        char c = expr_[pos_];
        
        // Number
        if (std::isdigit(c) || c == '.') {
            return parseNumber();
        }
        
        // Variable
        if (c == 'x' || c == 'y' || c == 'z' || c == 'r') {
            return parseVariable(c, x, y, z);
        }
        
        // Function call
        if (std::isalpha(c)) {
            return parseFunction(x, y, z);
        }
        
        // Parenthesized expression
        if (c == '(') {
            pos_++;  // Skip '('
            float result = parseExpression(x, y, z);
            skipWhitespace();
            if (pos_ < expr_.length() && expr_[pos_] == ')') {
                pos_++;  // Skip ')'
            }
            return result;
        }
        
        // Unary minus
        if (c == '-') {
            pos_++;
            return -parseFactor(x, y, z);
        }
        
        return 0.0f;
    }
    
    // Parse a term (factors connected by * or /)
    float parseTerm(float x, float y, float z) {
        float result = parseFactor(x, y, z);
        
        skipWhitespace();
        while (pos_ < expr_.length()) {
            char op = expr_[pos_];
            if (op == '*') {
                pos_++;
                result *= parseFactor(x, y, z);
            } else if (op == '/') {
                pos_++;
                float divisor = parseFactor(x, y, z);
                if (std::abs(divisor) > 1e-10f) {
                    result /= divisor;
                }
            } else if (op == '^') {
                pos_++;
                float exponent = parseFactor(x, y, z);
                result = std::pow(result, exponent);
            } else {
                break;
            }
            skipWhitespace();
        }
        
        return result;
    }
    
    // Parse an expression (terms connected by + or -)
    float parseExpression(float x, float y, float z) {
        float result = parseTerm(x, y, z);
        
        skipWhitespace();
        while (pos_ < expr_.length()) {
            char op = expr_[pos_];
            if (op == '+') {
                pos_++;
                result += parseTerm(x, y, z);
            } else if (op == '-') {
                pos_++;
                result -= parseTerm(x, y, z);
            } else {
                break;
            }
            skipWhitespace();
        }
        
        return result;
    }
    
public:
    SimpleExpressionParser(const std::string& expr) : expr_(expr), pos_(0) {}
    
    float evaluate(float x, float y, float z) {
        pos_ = 0;
        return parseExpression(x, y, z);
    }
};

std::vector<float> SignalGeneration::generateFromExpression(
    const std::vector<std::array<float, 3>>& points,
    const std::string& expression
) {
    if (expression.empty()) {
        return std::vector<float>(points.size(), 0.0f);
    }
    
    std::vector<float> values;
    values.reserve(points.size());
    
    try {
        SimpleExpressionParser parser(expression);
        
        for (const auto& point : points) {
            float value = parser.evaluate(point[0], point[1], point[2]);
            values.push_back(value);
        }
    } catch (...) {
        // On error, return zeros
        return std::vector<float>(points.size(), 0.0f);
    }
    
    return values;
}

} // namespace processing
} // namespace am_qadf_native
