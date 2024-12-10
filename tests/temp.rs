use moseiik::main::compute_mosaic;
use moseiik::main::Options;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_x86() {
        let args = Options {
            image: "assets/kit.jpeg".to_string(),
            tiles: "assets/tiles-small/".to_string(),
            output: "output_x86.png".to_string(),
            tile_size: 5,
            scaling: 1,
            remove_used: false,
            verbose: false,
            simd: true, // active l'utilisation de sse ou avx2
            num_thread: 1,
        };

        compute_mosaic(args); //Génère l'image faite avec les Tiles

        let generated = image::open("output_x86.png").expect("Failed to open generated image");
        let ground_truth = image::open("expected.png").expect("Failed to open expected image");

        assert_eq!(
            generated, ground_truth,
            "Generated image does not match the expect one!"
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_aarch64() {
        let args = Options {
            image: "assets/kit.jpeg".to_string(),
            tiles: "assets/tiles-small/".to_string(),
            output: "output_aarch64.png".to_string(),
            tile_size: 5,
            scaling: 1,
            remove_used: false,
            verbose: false,
            simd: true, // SIMD spécifique à ARM
            num_thread: 1,
        };

        compute_mosaic(args); //Génère l'image faite avec les Tiles

        let generated = image::open("output_aarch64.png").expect("Failed to open generated image");
        let ground_truth = image::open("expected.png").expect("Failed to open the expected image");

        assert_eq!(
            generated, ground_truth,
            "Generated image does not match!"
        );
    }

    #[test]
    fn test_generic() {
        let args = Options {
            image: "assets/kit.jpeg".to_string(),
            tiles: "assets/tiles-small/".to_string(),
            output: "output_generic.png".to_string(),
            tile_size: 5,
            scaling: 1,
            remove_used: false,
            verbose: false,
            simd: false, //active le mode générique
            num_thread: 1,
        };

        compute_mosaic(args); //Génère l'image faite avec les Tiles

        let generated = image::open("output_generic.png").expect("Failed to open generated image");
        let ground_truth = image::open("expected.png").expect("Failed to open the expected image");

        assert_eq!(
            generated, ground_truth,
            "Generated image does not match!"
        );
    }
}
