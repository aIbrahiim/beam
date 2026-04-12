/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.beam.sdk.schemas;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.time.LocalDate;
import java.time.LocalTime;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import org.apache.beam.sdk.coders.BigDecimalCoder;
import org.apache.beam.sdk.coders.BigEndianShortCoder;
import org.apache.beam.sdk.coders.BooleanCoder;
import org.apache.beam.sdk.coders.ByteArrayCoder;
import org.apache.beam.sdk.coders.ByteCoder;
import org.apache.beam.sdk.coders.Coder;
import org.apache.beam.sdk.coders.CoderException;
import org.apache.beam.sdk.coders.DoubleCoder;
import org.apache.beam.sdk.coders.FloatCoder;
import org.apache.beam.sdk.coders.InstantCoder;
import org.apache.beam.sdk.coders.IterableCoder;
import org.apache.beam.sdk.coders.ListCoder;
import org.apache.beam.sdk.coders.MapCoder;
import org.apache.beam.sdk.coders.NullableCoder;
import org.apache.beam.sdk.coders.StringUtf8Coder;
import org.apache.beam.sdk.coders.VarIntCoder;
import org.apache.beam.sdk.coders.VarLongCoder;
import org.apache.beam.sdk.schemas.Schema.FieldType;
import org.apache.beam.sdk.schemas.Schema.LogicalType;
import org.apache.beam.sdk.schemas.Schema.TypeName;
import org.apache.beam.sdk.util.common.ElementByteSizeObserver;
import org.apache.beam.vendor.guava.v32_1_2_jre.com.google.common.base.Preconditions;
import org.apache.beam.vendor.guava.v32_1_2_jre.com.google.common.collect.ImmutableMap;
import org.joda.time.ReadableInstant;

@SuppressWarnings({
  "nullness", // TODO(https://github.com/apache/beam/issues/20497)
  "rawtypes"
})
class SchemaCoderHelpers {
  // This contains a map of primitive types to their coders.
  private static final Map<TypeName, Coder> CODER_MAP =
      ImmutableMap.<TypeName, Coder>builder()
          .put(TypeName.BYTE, ByteCoder.of())
          .put(TypeName.BYTES, ByteArrayCoder.of())
          .put(TypeName.INT16, BigEndianShortCoder.of())
          .put(TypeName.INT32, VarIntCoder.of())
          .put(TypeName.INT64, VarLongCoder.of())
          .put(TypeName.DECIMAL, BigDecimalCoder.of())
          .put(TypeName.FLOAT, FloatCoder.of())
          .put(TypeName.DOUBLE, DoubleCoder.of())
          .put(TypeName.STRING, StringUtf8Coder.of())
          .put(TypeName.DATETIME, InstantCoder.of())
          .put(TypeName.BOOLEAN, BooleanCoder.of())
          .build();

  private static class LogicalTypeCoder<InputT, BaseT> extends Coder<InputT> {
    private final LogicalType<InputT, BaseT> logicalType;
    private final Coder<BaseT> baseTypeCoder;
    private final boolean isDateTime;

    LogicalTypeCoder(LogicalType<InputT, BaseT> logicalType, Coder baseTypeCoder) {
      this.logicalType = logicalType;
      this.baseTypeCoder = baseTypeCoder;
      this.isDateTime = logicalType.getBaseType().equals(FieldType.DATETIME);
    }

    /**
     * {@link org.apache.beam.sdk.schemas.logicaltypes.UnknownLogicalType} (and other pass-through
     * logical types with an INT64 representation) may leave {@link LocalDate} / {@link LocalTime}
     * values unchanged in {@link LogicalType#toBaseType}, while the wire coder expects {@link
     * Long}. Coerce so portable runners can encode rows built with standard SQL date/time
     * semantics.
     */
    /**
     * {@link #coderForFieldType} wraps nullable primitives in {@link NullableCoder}. Logical types
     * whose base is nullable {@code INT64} therefore use {@code NullableCoder(VarLongCoder)} on the
     * wire, and we must still coerce {@link LocalDate} / {@link LocalTime} to {@link Long}.
     */
    private static boolean unwrapsToVarLongCoder(Coder<?> coder) {
      Coder<?> current = coder;
      while (current instanceof NullableCoder) {
        current = ((NullableCoder<?>) current).getValueCoder();
      }
      return current instanceof VarLongCoder;
    }

    @SuppressWarnings("unchecked")
    private BaseT coerceJavaTimeForInt64Wire(BaseT baseOrInput) {
      if (!unwrapsToVarLongCoder(baseTypeCoder)) {
        return baseOrInput;
      }
      // Use TypeName, not FieldType.equals(INT64): nullable INT64 base (e.g. FieldType.INT64
      // with nullable=true) must still coerce LocalDate/LocalTime for VarLongCoder.
      if (logicalType.getBaseType().getTypeName() != TypeName.INT64) {
        return baseOrInput;
      }
      Object v = baseOrInput;
      if (v instanceof LocalDate) {
        return (BaseT) (Long) ((LocalDate) v).toEpochDay();
      }
      if (v instanceof LocalTime) {
        return (BaseT) (Long) ((LocalTime) v).toNanoOfDay();
      }
      return baseOrInput;
    }

    private BaseT toWireBaseType(InputT value) {
      BaseT baseType = logicalType.toBaseType(value);
      if (isDateTime) {
        baseType = (BaseT) ((ReadableInstant) baseType).toInstant();
      }
      return coerceJavaTimeForInt64Wire(baseType);
    }

    @Override
    public void encode(InputT value, OutputStream outStream) throws CoderException, IOException {
      baseTypeCoder.encode(toWireBaseType(value), outStream);
    }

    @Override
    public InputT decode(InputStream inStream) throws CoderException, IOException {
      BaseT baseType = baseTypeCoder.decode(inStream);
      return logicalType.toInputType(baseType);
    }

    @Override
    public List<? extends Coder<?>> getCoderArguments() {
      return Collections.emptyList();
    }

    @Override
    public void verifyDeterministic() throws NonDeterministicException {
      baseTypeCoder.verifyDeterministic();
    }

    @Override
    public boolean consistentWithEquals() {
      // we can't assume that InputT is consistent with equals.
      // TODO: We should plumb this through to logical types.
      return false;
    }

    @Override
    public Object structuralValue(InputT value) {
      BaseT wireBase = toWireBaseType(value);
      if (baseTypeCoder.consistentWithEquals()) {
        return wireBase;
      } else {
        return baseTypeCoder.structuralValue(wireBase);
      }
    }

    @Override
    public boolean isRegisterByteSizeObserverCheap(InputT value) {
      return baseTypeCoder.isRegisterByteSizeObserverCheap(toWireBaseType(value));
    }

    @Override
    public void registerByteSizeObserver(InputT value, ElementByteSizeObserver observer)
        throws Exception {
      baseTypeCoder.registerByteSizeObserver(toWireBaseType(value), observer);
    }
  }

  /** Returns the coder used for a given primitive type. */
  public static <T> Coder<T> coderForFieldType(FieldType fieldType) {
    Coder<T> coder;
    switch (fieldType.getTypeName()) {
      case ROW:
        coder = (Coder<T>) SchemaCoder.of(fieldType.getRowSchema());
        break;
      case ARRAY:
        coder = (Coder<T>) ListCoder.of(coderForFieldType(fieldType.getCollectionElementType()));
        break;
      case ITERABLE:
        coder =
            (Coder<T>) IterableCoder.of(coderForFieldType(fieldType.getCollectionElementType()));
        break;
      case MAP:
        coder =
            (Coder<T>)
                MapCoder.of(
                    coderForFieldType(fieldType.getMapKeyType()),
                    coderForFieldType(fieldType.getMapValueType()));
        break;
      case LOGICAL_TYPE:
        coder =
            new LogicalTypeCoder(
                fieldType.getLogicalType(),
                coderForFieldType(fieldType.getLogicalType().getBaseType()));
        break;
      default:
        coder = (Coder<T>) CODER_MAP.get(fieldType.getTypeName());
    }
    Preconditions.checkNotNull(coder, "Unexpected field type %s", fieldType.getTypeName());
    if (fieldType.getNullable()) {
      coder = NullableCoder.of(coder);
    }
    return coder;
  }
}
