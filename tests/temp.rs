#[cfg(test)]
mod tests {
    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_x86() {
        // TODO
        // test avx2 or sse2 if available
        assert!(true);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_aarch64() {
        //TODO
        assert!(true);
    }

    #[test]
    fn test_generic() {
        //TODO
        assert!(true);
    }
}
