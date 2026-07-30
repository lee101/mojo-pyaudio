"""Frame-agnostic byte ring operations for interleaved PCM streams."""

from std.sys import simd_width_of

comptime BytePtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()
comptime COPY_WIDTH = 8 * W


def copy_bytes_serial(source: BytePtr, destination: BytePtr, count: Int):
    var index = 0
    while index + COPY_WIDTH <= count:
        destination.store(index, source.load[width=COPY_WIDTH](index))
        index += COPY_WIDTH
    while index < count:
        destination[index] = source[index]
        index += 1


def copy_bytes(source_addr: Int, destination_addr: Int, count: Int):
    copy_bytes_serial(
        BytePtr(unsafe_from_address=source_addr),
        BytePtr(unsafe_from_address=destination_addr),
        count,
    )


def valid_ring(capacity: Int, position: Int, count: Int) -> Bool:
    return capacity > 0 and position >= 0 and position < capacity and count >= 0 and count <= capacity


@export("mpa_copy")
def mpa_copy(source_addr: Int, destination_addr: Int, count: Int) abi("C") -> Int:
    if count < 0:
        return -1
    if count == 0:
        return 0
    if source_addr == 0 or destination_addr == 0:
        return -1
    copy_bytes(source_addr, destination_addr, count)
    return 0


@export("mpa_ring_write")
def mpa_ring_write(
    ring_addr: Int,
    capacity: Int,
    position: Int,
    source_addr: Int,
    count: Int,
) abi("C") -> Int:
    if not valid_ring(capacity, position, count):
        return -1
    if count == 0:
        return position
    if ring_addr == 0 or source_addr == 0:
        return -1

    var first = min(count, capacity - position)
    copy_bytes(source_addr, ring_addr + position, first)
    if first < count:
        copy_bytes(source_addr + first, ring_addr, count - first)
    return (position + count) % capacity


@export("mpa_ring_read")
def mpa_ring_read(
    ring_addr: Int,
    capacity: Int,
    position: Int,
    destination_addr: Int,
    count: Int,
) abi("C") -> Int:
    if not valid_ring(capacity, position, count):
        return -1
    if count == 0:
        return position
    if ring_addr == 0 or destination_addr == 0:
        return -1

    var first = min(count, capacity - position)
    copy_bytes(ring_addr + position, destination_addr, first)
    if first < count:
        copy_bytes(ring_addr, destination_addr + first, count - first)
    return (position + count) % capacity


@export("mpa_ring_transfer")
def mpa_ring_transfer(
    source_ring_addr: Int,
    source_capacity: Int,
    source_position: Int,
    destination_ring_addr: Int,
    destination_capacity: Int,
    destination_position: Int,
    count: Int,
) abi("C") -> Int:
    if (
        not valid_ring(source_capacity, source_position, count)
        or not valid_ring(destination_capacity, destination_position, count)
        or source_ring_addr == 0
        or destination_ring_addr == 0
    ):
        return -1
    if count == 0:
        return destination_position

    var source_index = source_position
    var destination_index = destination_position
    var remaining = count
    while remaining > 0:
        var chunk = min(
            remaining,
            min(source_capacity - source_index, destination_capacity - destination_index),
        )
        copy_bytes(
            source_ring_addr + source_index,
            destination_ring_addr + destination_index,
            chunk,
        )
        remaining -= chunk
        source_index = (source_index + chunk) % source_capacity
        destination_index = (destination_index + chunk) % destination_capacity
    return destination_index
