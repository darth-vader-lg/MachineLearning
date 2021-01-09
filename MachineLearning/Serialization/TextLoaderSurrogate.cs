using Microsoft.ML.Data;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   /// <summary>
   /// Classe helper per la serializzazione dei dati del textLoader
   /// </summary>
   internal class TextLoaderSurrogate
   {
      internal class Column : ISerializationSurrogate<TextLoader.Column>
      {
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            var data = (TextLoader.Column)obj;
            info.AddValue(nameof(data.DataKind), (byte)data.DataKind);
            info.AddValue($"{nameof(data.KeyCount)}.Exists", data.KeyCount != null);
            info.AddValue(nameof(data.KeyCount), data.KeyCount?.Count);
            info.AddValue(nameof(data.Name), data.Name);
            info.AddValue(nameof(data.Source), data.Source);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = (TextLoader.Column)obj;
            data.DataKind = (DataKind)info.GetValue(nameof(data.DataKind), typeof(byte));
            if (info.GetBoolean($"{nameof(data.KeyCount)}.Exists"))
               data.KeyCount = new KeyCount { Count = (ulong?)info.GetValue(nameof(data.KeyCount), typeof(ulong?)) };
            data.Name = (string)info.GetValue(nameof(data.Name), typeof(string));
            data.Source = (TextLoader.Range[])info.GetValue(nameof(data.Source), typeof(TextLoader.Range[]));
            return data;
         }
      }
      internal class Options : ISerializationSurrogate<TextLoader.Options>
      {
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            var data = (TextLoader.Options)obj;
            info.AddValue(nameof(data.AllowQuoting), data.AllowQuoting);
            info.AddValue(nameof(data.AllowSparse), data.AllowSparse);
            info.AddValue(nameof(data.Columns), data.Columns);
            info.AddValue(nameof(data.DecimalMarker), data.DecimalMarker);
            info.AddValue(nameof(data.EscapeChar), data.EscapeChar);
            info.AddValue(nameof(data.HasHeader), data.HasHeader);
            info.AddValue(nameof(data.HeaderFile), data.HeaderFile);
            info.AddValue(nameof(data.InputSize), data.InputSize);
            info.AddValue(nameof(data.MaxRows), data.MaxRows);
            info.AddValue(nameof(data.MissingRealsAsNaNs), data.MissingRealsAsNaNs);
            info.AddValue(nameof(data.ReadMultilines), data.ReadMultilines);
            info.AddValue(nameof(data.Separators), data.Separators);
            info.AddValue(nameof(data.TrimWhitespace), data.TrimWhitespace);
            info.AddValue(nameof(data.UseThreads), data.UseThreads);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = (TextLoader.Options)obj;
            data.AllowQuoting = (bool)info.GetValue(nameof(data.AllowQuoting), typeof(bool));
            data.AllowSparse = (bool)info.GetValue(nameof(data.AllowSparse), typeof(bool));
            data.Columns = (TextLoader.Column[])info.GetValue(nameof(data.Columns), typeof(TextLoader.Column[]));
            data.DecimalMarker = (char)info.GetValue(nameof(data.DecimalMarker), typeof(char));
            data.EscapeChar = (char)info.GetValue(nameof(data.EscapeChar), typeof(char));
            data.HasHeader = (bool)info.GetValue(nameof(data.HasHeader), typeof(bool));
            data.HeaderFile = (string)info.GetValue(nameof(data.HeaderFile), typeof(string));
            data.InputSize = (int?)info.GetValue(nameof(data.InputSize), typeof(int?));
            data.MaxRows = (long?)info.GetValue(nameof(data.MaxRows), typeof(long?));
            data.MissingRealsAsNaNs = (bool)info.GetValue(nameof(data.MissingRealsAsNaNs), typeof(bool));
            data.ReadMultilines = (bool)info.GetValue(nameof(data.ReadMultilines), typeof(bool));
            data.Separators = (char[])info.GetValue(nameof(data.Separators), typeof(char[]));
            data.TrimWhitespace = (bool)info.GetValue(nameof(data.TrimWhitespace), typeof(bool));
            data.UseThreads = (bool)info.GetValue(nameof(data.UseThreads), typeof(bool));
            return data;
         }
      }

      internal class Range : ISerializationSurrogate<TextLoader.Range>
      {
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            var data = (TextLoader.Range)obj;
            info.AddValue(nameof(data.AllOther), data.AllOther);
            info.AddValue(nameof(data.AutoEnd), data.AutoEnd);
            info.AddValue(nameof(data.ForceVector), data.ForceVector);
            info.AddValue(nameof(data.Max), data.Max);
            info.AddValue(nameof(data.Min), data.Min);
            info.AddValue(nameof(data.VariableEnd), data.VariableEnd);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = (TextLoader.Range)obj;
            data.AllOther = (bool)info.GetValue(nameof(data.AllOther), typeof(bool));
            data.AutoEnd = (bool)info.GetValue(nameof(data.AutoEnd), typeof(bool));
            data.ForceVector = (bool)info.GetValue(nameof(data.ForceVector), typeof(bool));
            data.Max = (int?)info.GetValue(nameof(data.Max), typeof(int?));
            data.Min = (int)info.GetValue(nameof(data.Min), typeof(int));
            data.VariableEnd = (bool)info.GetValue(nameof(data.VariableEnd), typeof(bool));
            return data;
         }
      }
   }
}
