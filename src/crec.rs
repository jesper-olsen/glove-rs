use bytemuck::pod_read_unaligned;
use bytemuck::{Pod, Zeroable};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::cmp::Ordering;
use std::io::{self, ErrorKind, Read, Write};
use std::mem;

/// Co-occurrence record struct. `repr(C)` and `Pod` ensure the memory layout
/// is identical to the C struct, allowing us to read the binary file directly.
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
#[repr(C)]
pub struct Crec {
    pub word1: u32,
    pub word2: u32,
    pub val: f64,
}

impl Ord for Crec {
    fn cmp(&self, other: &Self) -> Ordering {
        // sort order: word 1 then word 2
        self.word1
            .cmp(&other.word1)
            .then_with(|| self.word2.cmp(&other.word2))
    }
}

impl PartialOrd for Crec {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// Manual implementation of PartialEq.
// We ONLY compare word1 and word2, ignoring `val`.
impl PartialEq for Crec {
    fn eq(&self, other: &Self) -> bool {
        self.word1 == other.word1 && self.word2 == other.word2
    }
}

impl Eq for Crec {} // Marker trait - necessary for Ord

impl Crec {
    // write one CREC to a binary stream (File or Stdout)
    pub fn write_to<W: Write>(writer: &mut W, crec: &Crec) -> io::Result<()> {
        writer.write_u32::<LittleEndian>(crec.word1)?;
        writer.write_u32::<LittleEndian>(crec.word2)?;
        writer.write_f64::<LittleEndian>(crec.val)?;
        Ok(())
    }

    /// Distinguishes between clean EOF and other I/O errors.
    /// - `Ok(Some(crec))`: Success.
    /// - `Ok(None)`: Clean End-Of-File.
    /// - `Err(e)`: An I/O error occurred.
    pub fn read_from<R: Read>(reader: &mut R) -> io::Result<Option<Self>> {
        let word1 = match reader.read_u32::<LittleEndian>() {
            Ok(val) => val,
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e),
        };
        let word2 = reader.read_u32::<LittleEndian>()?;
        let val = reader.read_f64::<LittleEndian>()?;
        Ok(Some(Crec { word1, word2, val }))
    }

    /// The _raw methods for reading and writing take the endianness for given
    /// Don't use raw if training and testing on opposite endian systems...

    /// Distinguishes between clean EOF and other I/O errors.
    /// - `Ok(Some(crec))`: Success.
    /// - `Ok(None)`: Clean End-Of-File.
    /// - `Err(e)`: An I/O error occurred.
    pub fn read_from_raw<R: Read>(reader: &mut R) -> io::Result<Option<Self>> {
        let mut buffer = [0u8; mem::size_of::<Crec>()];
        match reader.read_exact(&mut buffer) {
            Ok(()) => {
                let crec = pod_read_unaligned(&buffer);
                Ok(Some(crec))
            }
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Writes a single Crec record to a writer.
    pub fn write_to_raw<W: Write>(writer: &mut W, crec: &Crec) -> io::Result<()> {
        // Convert Crec to a byte slice safely using bytemuck
        let buffer = bytemuck::bytes_of(crec);
        writer.write_all(buffer)
    }

    /// Writes a slice of Crec records to a writer in a single, efficient operation.
    pub fn write_slice_raw<W: Write>(writer: &mut W, crecs: &[Crec]) -> io::Result<()> {
        // `unsafe` because we are asserting that the memory layout of a slice
        // of `Crec`s is equivalent to a contiguous slice of bytes.
        let byte_slice = unsafe {
            std::slice::from_raw_parts(crecs.as_ptr() as *const u8, std::mem::size_of_val(crecs))
        };
        writer.write_all(byte_slice)
    }
}
